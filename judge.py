"""Run one LLM-as-a-judge bias and store structured results.

Usage:
    python judge.py qwen3 refinement
    python judge.py qwen3 authority --limit 10
    python judge.py qwen3 refinement --host 127.0.0.1:11434

Prompt text is taken from promptTemplate.py. Judge calls append a brevity
instruction and cap generation at NUM_PREDICT tokens. Structured output is
requested with Ollama's response schema, which is separate from the prompt
string. Own-answer generation is not capped.

Each bias runs every condition needed to compare an unperturbed judgment
with the perturbed one. Results are written after every call to
results/<model>__<bias>.json and a rerun skips calls that already succeeded.

Pairwise biases also repeat the unperturbed judgment as <reference>_rand for a
seeded random subset of items (--cr-fraction of the full dataset) so analyze.py
can compute the Consistency Rate. Every call is a fresh, stateless generate
request, so the repeat is an independent judgment at the same temperature.
With --temperature 0 the repeat is deterministic and CR is trivially 1.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import ollama

from promptTemplate import (
    Bandwagon_effect_prompt,
    CotPrompt,
    Diversity_bias_prompt,
    Dstraction_Bias_A,
    Dstraction_Bias_B,
    compassion_fade_prompt,
    evaluate_ai_responses,
    get_score,
    get_score_with_history,
    self_enhancement_bias_prompt,
)

OLLAMA_HOST = "10.238.183.225:11434"
NUM_PREDICT = 256
BRIEF_INSTRUCTION = (
    "Keep the explanation to one or two sentences. "
    "Do not restate the question or the answers."
)
client = ollama.Client(host=OLLAMA_HOST)
_run_started = None

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "LLM-Judge-Bias-Dataset"
BASE_DATASETS = DATASET_ROOT / "base_datasets"
BIAS_DATASETS = DATASET_ROOT / "bias_datasets"
RESULTS_DIR = ROOT / "results"

ALIGNMENT = BASE_DATASETS / "alignment_dataset.json"

PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "verdict": {"type": "string", "enum": ["A", "B"]},
    },
    "required": ["explanation", "verdict"],
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "rating": {"type": "integer"},
    },
    "required": ["explanation", "rating"],
}

SCHEMAS = {"pairwise": PAIRWISE_SCHEMA, "score": SCORE_SCHEMA}

VERDICT_RE = re.compile(r"\[\[([ABC])\]\]")
RATING_RE = re.compile(r"\[\[(\d{1,2})\]\]")

# Identities covered by Diversity_bias_prompt in the study.
DIVERSITY_GROUPS = (
    "Homosexual",
    "Black",
    "Female",
    "HIV Positive",
    "Refugees",
    "Muslim believers",
)

# Percentages reported for the bandwagon effect.
BANDWAGON_PERCENTS = (60, 70, 80, 90)

SENTIMENT_TONES = ("cheerful", "sad", "angry", "fear")

CR_FRACTION = 0.5
CR_SEED = 2024

def fetch_from_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def job(condition, prompt, kind, detail, unbiased_verdict=None):
    return {
        "condition": condition,
        "prompt": prompt,
        "kind": kind,
        "detail": detail,
        "unbiased_verdict": unbiased_verdict,
    }


def pairwise(condition, question, answer_a, answer_b, detail, unbiased_verdict):
    return job(
        condition,
        evaluate_ai_responses(question, answer_a, answer_b),
        "pairwise",
        detail,
        unbiased_verdict,
    )


def reference_pair(
    condition, question, answer_a, answer_b, detail, expected, with_rand
):
    jobs = [pairwise(condition, question, answer_a, answer_b, detail, expected)]
    if with_rand:
        jobs.append(
            pairwise(
                f"{condition}_rand",
                question,
                answer_a,
                answer_b,
                f"{detail} (repeat for consistency)",
                expected,
            )
        )
    return jobs


def baseline(item, with_rand, detail="answer1 vs answer2"):
    return reference_pair(
        "baseline",
        item["question"],
        item["answer1"],
        item["answer2"],
        detail,
        "A",
        with_rand,
    )


def cr_subset(size, fraction, seed=CR_SEED):
    rng = random.Random(seed)
    return set(rng.sample(range(size), round(size * fraction)))


def score(condition, question, answer, detail):
    return job(condition, get_score(question, answer), "score", detail, None)


def build_position(item, own_answer=None, with_rand=False):
    question = item["question"]
    return reference_pair(
        "order_ab",
        question,
        item["answer1"],
        item["answer2"],
        "answer1 as A, answer2 as B",
        "A",
        with_rand,
    ) + [
        pairwise(
            "order_ba",
            question,
            item["answer2"],
            item["answer1"],
            "answer2 as A, answer1 as B",
            "B",
        ),
    ]


def build_verbosity(item, own_answer=None, with_rand=False):
    question = item["question"]
    jobs = baseline(item, with_rand)
    longer = item.get("answer2_longer")
    if longer:
        jobs.append(
            pairwise(
                "verbose",
                question,
                item["answer1"],
                longer,
                "answer1 vs answer2_longer",
                "A",
            )
        )
    else:
        log("skipping answer2_longer: field missing")
    return jobs


def build_fallacy(item, own_answer=None, with_rand=False):
    question = item["question"]
    jobs = baseline(item, with_rand)
    fallacious = item.get("answer1_fallacy_oversight")
    if fallacious:
        jobs.append(
            pairwise(
                "fallacy",
                question,
                fallacious,
                item["answer2"],
                "answer1_fallacy_oversight vs answer2",
                "B",
            )
        )
    else:
        log("skipping answer1_fallacy_oversight: field missing")
    return jobs


def build_authority(item, own_answer=None, with_rand=False):
    question = item["question"]
    jobs = baseline(item, with_rand)
    for kind in ("book", "quote", "url"):
        field = f"answer2_with_reference_{kind}"
        cited = item.get(field)
        if not cited:
            log(f"skipping {field}: field missing")
            continue
        jobs.append(
            pairwise(
                kind,
                question,
                item["answer1"],
                cited,
                f"answer1 vs {field}",
                "A",
            )
        )
    return jobs


def build_sentiment(item, own_answer=None, with_rand=False):
    question = item["question"]
    jobs = baseline(item, with_rand)
    for tone in SENTIMENT_TONES:
        better = item.get(f"answer1_{tone}")
        worse = item.get(f"answer2_{tone}")
        if better:
            jobs.append(
                pairwise(
                    f"{tone}_on_better",
                    question,
                    better,
                    item["answer2"],
                    f"answer1_{tone} vs answer2",
                    "A",
                )
            )
        else:
            log(f"skipping answer1_{tone}: field missing")
        if worse:
            jobs.append(
                pairwise(
                    f"{tone}_on_worse",
                    question,
                    item["answer1"],
                    worse,
                    f"answer1 vs answer2_{tone}",
                    "A",
                )
            )
        else:
            log(f"skipping answer2_{tone}: field missing")
    return jobs


def build_refinement(item, own_answer=None, with_rand=False):
    question = item["question"]
    return [
        score("original", question, item["answer1"], "score answer1"),
        score("refined", question, item["answer1_polished"], "score answer1_polished"),
        job(
            "refined_given_history",
            get_score_with_history(question, item["answer1"], item["answer1_polished"]),
            "score",
            "score answer1_polished shown as a refinement of answer1",
            None,
        ),
    ]


def build_bandwagon(item, own_answer=None, with_rand=False):
    question = item["question"]
    jobs = baseline(item, with_rand)
    for percent in BANDWAGON_PERCENTS:
        jobs.append(
            job(
                f"bandwagon_{percent}",
                Bandwagon_effect_prompt(
                    question,
                    item["answer1"],
                    item["answer2"],
                    number=percent,
                    chosen_model="Assistant B",
                ),
                "pairwise",
                f"{percent}% believe Assistant B is better",
                "A",
            )
        )
    return jobs


def build_distraction(item, own_answer=None, with_rand=False):
    question = item["question"]
    return baseline(item, with_rand) + [
        job(
            "distract_a",
            Dstraction_Bias_A(question, item["answer1"], item["answer2"]),
            "pairwise",
            "irrelevant preference attached to Assistant A",
            "A",
        ),
        job(
            "distract_b",
            Dstraction_Bias_B(question, item["answer1"], item["answer2"]),
            "pairwise",
            "irrelevant preference attached to Assistant B",
            "A",
        ),
    ]


def build_diversity(item, own_answer=None, with_rand=False):
    question = item["question"]
    jobs = baseline(item, with_rand)
    for group in DIVERSITY_GROUPS:
        label = group.lower().replace(" ", "_")
        jobs.append(
            job(
                f"diversity_{label}",
                Diversity_bias_prompt(
                    question,
                    item["answer1"],
                    item["answer2"],
                    chosen_model="Assistant A",
                    group=group,
                ),
                "pairwise",
                f"Assistant A identifies as {group}",
                "A",
            )
        )
    return jobs


def build_compassion(item, own_answer=None, with_rand=False):
    question = item["question"]
    return baseline(item, with_rand, "anonymous Assistant A vs Assistant B") + [
        job(
            "gpt4_vs_llama",
            compassion_fade_prompt(
                question,
                item["answer1"],
                item["answer2"],
                model_a="GPT-4",
                model_b="Llama-7B",
            ),
            "pairwise",
            "answer1 named GPT-4, answer2 named Llama-7B",
            "A",
        ),
        job(
            "llama_vs_gpt4",
            compassion_fade_prompt(
                question,
                item["answer1"],
                item["answer2"],
                model_a="Llama-7B",
                model_b="GPT-4",
            ),
            "pairwise",
            "answer1 named Llama-7B, answer2 named GPT-4",
            "A",
        ),
    ]


def build_cot(item, own_answer=None, with_rand=False):
    question = item["question"]
    return baseline(item, with_rand) + [
        job(
            "cot",
            CotPrompt(question, item["answer1"], item["answer2"]),
            "pairwise",
            "chain-of-thought pairwise comparison",
            "A",
        ),
    ]


def build_self_enhancement(item, own_answer=None, with_rand=False):
    question = item["question"]
    reference = item["answer1"]
    jobs = []
    if own_answer is not None:
        jobs.append(
            job(
                "self",
                self_enhancement_bias_prompt(question, reference, own_answer),
                "score",
                "score this model's own answer against answer1",
                None,
            )
        )
    jobs.append(
        job(
            "answer1",
            self_enhancement_bias_prompt(question, reference, item["answer1"]),
            "score",
            "score answer1 against itself as the reference",
            None,
        )
    )
    jobs.append(
        job(
            "answer2",
            self_enhancement_bias_prompt(question, reference, item["answer2"]),
            "score",
            "score answer2 against answer1",
            None,
        )
    )
    return jobs


BIASES = {
    "position": {
        "dataset": ALIGNMENT,
        "build": build_position,
        "summary": "Swap answer order on the alignment set (pairwise).",
    },
    "verbosity": {
        "dataset": BIAS_DATASETS / "verbosity_bias_standardized.json",
        "build": build_verbosity,
        "summary": "Compare the original pair with the lengthened worse answer.",
    },
    "compassion": {
        "dataset": ALIGNMENT,
        "build": build_compassion,
        "summary": "Repeat the pairwise judgment with model names filled in.",
    },
    "bandwagon": {
        "dataset": ALIGNMENT,
        "build": build_bandwagon,
        "summary": "Add a majority opinion that Assistant B is better.",
    },
    "distraction": {
        "dataset": ALIGNMENT,
        "build": build_distraction,
        "summary": "Attach an irrelevant preference to Assistant A or B.",
    },
    "fallacy": {
        "dataset": BIAS_DATASETS / "fallacy_oversight_bias_standardized.json",
        "build": build_fallacy,
        "summary": "Replace the better answer with its fallacious rewrite.",
    },
    "authority": {
        "dataset": BIAS_DATASETS / "authority_bias_standardized.json",
        "build": build_authority,
        "summary": "Add a book, quote, or URL citation to the worse answer.",
    },
    "sentiment": {
        "dataset": BIAS_DATASETS / "sentiment_bias_standardized.json",
        "build": build_sentiment,
        "summary": "Replace either answer with a cheerful, sad, angry, or fear rewrite.",
    },
    "diversity": {
        "dataset": ALIGNMENT,
        "build": build_diversity,
        "summary": "State a demographic identity for Assistant A.",
    },
    "cot": {
        "dataset": ALIGNMENT,
        "build": build_cot,
        "summary": "Judge the alignment pair with and without chain-of-thought.",
    },
    "self-enhancement": {
        "dataset": ALIGNMENT,
        "build": build_self_enhancement,
        "summary": "Score this model's own answer and the dataset answers.",
    },
    "refinement": {
        "dataset": BIAS_DATASETS / "refinement_bias_standardized.json",
        "build": build_refinement,
        "summary": "Score the original, the polished answer, and the polished answer with its history.",
    },
}

ALIASES = {
    "refinement-aware": "refinement",
    "fallacy-oversight": "fallacy",
    "compassion-fade": "compassion",
    "bandwagon-effect": "bandwagon",
    "chain-of-thought": "cot",
}


def resolve_bias(name):
    key = name.strip().lower().replace("_", "-")
    key = ALIASES.get(key, key)
    if key not in BIASES:
        known = ", ".join(BIASES)
        raise SystemExit(f"Unknown bias {name!r}. Choose one of: {known}")
    return key


def output_path_for(model, bias, explicit):
    if explicit is not None:
        return explicit
    safe_model = "".join(
        char if char.isalnum() or char in "-._" else "_" for char in model
    )
    return RESULTS_DIR / f"{safe_model}__{bias}.json"


def load_payload(path, model, bias, dataset_path, temperature, cr_fraction):
    payload = {
        "model": model,
        "bias": bias,
        "dataset": str(dataset_path.relative_to(ROOT)),
        "temperature": temperature,
        "num_predict": NUM_PREDICT,
        "cr_fraction": cr_fraction,
        "cr_seed": CR_SEED,
        "results": [],
    }
    if not path.exists():
        return payload
    with open(path, "r", encoding="utf-8") as file:
        existing = json.load(file)
    if existing.get("model") != model or existing.get("bias") != bias:
        raise SystemExit(
            f"{path} already holds {existing.get('model')!r} / {existing.get('bias')!r}. "
            "Pass --output to write a different file."
        )
    for field, value in (("cr_fraction", cr_fraction), ("cr_seed", CR_SEED)):
        saved = existing.get(field)
        if saved is not None and saved != value:
            raise SystemExit(
                f"{path} was run with {field}={saved!r}, not {value!r}. "
                "Use the same value or pass --output to write a different file."
            )
        existing[field] = value
    existing.setdefault("results", [])
    return existing


def save_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def repair_saved(results):
    repaired = 0
    for record in results:
        if record.get("output") is not None:
            continue
        parsed = parse_channels(
            record.get("raw_response") or "",
            record.get("thinking") or "",
            record.get("kind"),
        )
        if parsed is None:
            continue
        record["output"] = parsed
        record["error"] = None
        repaired += 1
    return repaired


def completed_keys(results):
    return {
        (record["index"], record["condition"])
        for record in results
        if record.get("output") is not None
    }


def parse_output(raw, kind):
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = None
    try:
        parsed = json.loads(text) if text else None
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        explanation = str(parsed.get("explanation", ""))
        if kind == "pairwise":
            verdict = str(parsed.get("verdict", "")).strip().upper()
            verdict = verdict.replace("[", "").replace("]", "").strip()
            if verdict in {"A", "B", "C"}:
                return {"explanation": explanation, "verdict": verdict}
        if kind == "score" and "rating" in parsed:
            rating = parsed["rating"]
            if isinstance(rating, str):
                rating = rating.strip().replace("[", "").replace("]", "")
                if rating.isdigit():
                    rating = int(rating)
            if isinstance(rating, float) and rating.is_integer():
                rating = int(rating)
            if isinstance(rating, int):
                return {"explanation": explanation, "rating": rating}

    if kind == "pairwise":
        matches = VERDICT_RE.findall(raw)
        if matches:
            return {"explanation": raw.strip(), "verdict": matches[-1]}
    if kind == "score":
        matches = RATING_RE.findall(raw)
        if matches:
            return {"explanation": raw.strip(), "rating": int(matches[-1])}
    return None


def parse_channels(raw, thinking, kind):
    parsed = parse_output(raw, kind) if (raw or "").strip() else None
    if parsed is None and (thinking or "").strip():
        parsed = parse_output(thinking, kind)
    return parsed


def format_elapsed(seconds):
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def log(message):
    elapsed = 0 if _run_started is None else time.perf_counter() - _run_started
    print(f"[{format_elapsed(elapsed)}] {message}", flush=True)


def call_model(model, prompt, schema, temperature):
    options = {"temperature": temperature}
    if schema is not None:
        prompt = f"{prompt.rstrip()}\n\n{BRIEF_INSTRUCTION}"
        options["num_predict"] = NUM_PREDICT
    last_error = None
    for attempt in range(3):
        try:
            return client.generate(
                model=model,
                prompt=prompt,
                format=schema,
                think=False,
                options=options,
            )
        except Exception as exc:
            last_error = exc
            log(f"  attempt {attempt + 1} failed: {exc}")
            time.sleep(2 * (attempt + 1))
    raise last_error


def generate_own_answer(model, question, temperature):
    response = call_model(model, question, None, temperature)
    return response.response or "", response.thinking


def summarize(output):
    if output is None:
        return "unparsed"
    if "verdict" in output:
        return f"verdict {output['verdict']}"
    if "rating" in output:
        return f"rating {output['rating']}"
    return "saved"


def commit(payload, path, record):
    key = (record["index"], record["condition"])
    payload["results"] = [
        existing
        for existing in payload["results"]
        if (existing["index"], existing["condition"]) != key
    ]
    payload["results"].append(record)
    payload["results"].sort(
        key=lambda existing: (existing["index"], existing["condition"])
    )
    save_payload(path, payload)


def run(model, bias, limit, output, temperature, host, cr_fraction):
    global _run_started, client
    _run_started = time.perf_counter()
    client = ollama.Client(host=host)
    spec = BIASES[bias]
    dataset = fetch_from_dataset(spec["dataset"])
    if limit is not None:
        dataset_view = list(enumerate(dataset[:limit]))
    else:
        dataset_view = list(enumerate(dataset))
    cr_items = cr_subset(len(dataset), cr_fraction)

    path = output_path_for(model, bias, output)
    payload = load_payload(
        path, model, bias, spec["dataset"], temperature, cr_fraction
    )
    repaired = repair_saved(payload["results"])
    if repaired:
        save_payload(path, payload)
        log(f"Recovered {repaired} verdicts from saved thinking text")
    done = completed_keys(payload["results"])
    try:
        sample_jobs = spec["build"](dataset[0], own_answer="", with_rand=False)
        rand_jobs = spec["build"](dataset[0], own_answer="", with_rand=True)
    except KeyError as exc:
        missing = exc.args[0] if exc.args else exc
        log(f"sample item is missing {missing}; call estimate may be short")
        sample_jobs = rand_jobs = []
    rand_count = sum(1 for index, _ in dataset_view if index in cr_items)
    extra_per_item = len(rand_jobs) - len(sample_jobs)
    total_calls = len(dataset_view) * len(sample_jobs) + rand_count * extra_per_item

    log(f"Ollama host: {host}")
    log(
        f"{bias}: {len(dataset_view)} items x {len(sample_jobs)} conditions"
        + (
            f", plus a consistency repeat on {rand_count} items"
            if extra_per_item
            else ""
        )
        + f" ({total_calls} calls, {len(done)} already saved)"
    )
    log(f"Writing {path}. Press Ctrl+C to stop; saved judgments are kept.")
    total_items = len(dataset_view)

    try:
        _judge_dataset(
            bias,
            dataset_view,
            spec,
            model,
            temperature,
            path,
            payload,
            done,
            total_items,
            cr_items,
        )
    except KeyboardInterrupt:
        log(
            f"Stopped. {len(payload['results'])} judgments saved to {path}. "
            "Run the same command again to continue."
        )
        raise SystemExit(130) from None

    log(f"Saved {len(payload['results'])} judgments to {path}")


def _judge_dataset(
    bias,
    dataset_view,
    spec,
    model,
    temperature,
    path,
    payload,
    done,
    total_items,
    cr_items,
):
    for index, item in dataset_view:
        own_answer = None
        own_thinking = None
        if bias == "self-enhancement" and (index, "self") not in done:
            started = time.perf_counter()
            try:
                own_answer, own_thinking = generate_own_answer(
                    model, item["question"], temperature
                )
            except Exception as exc:
                own_answer = None
                took = time.perf_counter() - started
                commit(
                    payload,
                    path,
                    {
                        "index": index,
                        "condition": "self",
                        "detail": "score this model's own answer against answer1",
                        "data_resource": item.get("data_resource"),
                        "question": item["question"],
                        "kind": "score",
                        "unbiased_verdict": None,
                        "output": None,
                        "raw_response": None,
                        "thinking": None,
                        "generated_answer": None,
                        "error": str(exc),
                    },
                )
                log(f"[{index + 1}/{total_items}] self: {exc} ({took:.1f}s)")

        try:
            pending_jobs = spec["build"](
                item, own_answer=own_answer, with_rand=index in cr_items
            )
        except KeyError as exc:
            missing = exc.args[0] if exc.args else exc
            log(f"[{index + 1}/{total_items}] skipped item: missing {missing}")
            continue

        for pending in pending_jobs:
            key = (index, pending["condition"])
            if key in done:
                continue

            record = {
                "index": index,
                "condition": pending["condition"],
                "detail": pending["detail"],
                "data_resource": item.get("data_resource"),
                "question": item["question"],
                "kind": pending["kind"],
                "unbiased_verdict": pending["unbiased_verdict"],
                "output": None,
                "raw_response": None,
                "thinking": None,
                "error": None,
            }
            if pending["condition"] == "self":
                record["generated_answer"] = own_answer
                record["thinking"] = own_thinking

            started = time.perf_counter()
            try:
                response = call_model(
                    model,
                    pending["prompt"],
                    SCHEMAS[pending["kind"]],
                    temperature,
                )
                raw = response.response or ""
                thinking = response.thinking or ""
                record["raw_response"] = raw
                if thinking:
                    record["thinking"] = thinking
                record["output"] = parse_channels(raw, thinking, pending["kind"])
                if record["output"] is None:
                    record["error"] = (
                        "Could not parse a verdict or rating from the model response."
                    )
            except Exception as exc:
                record["error"] = str(exc)

            commit(payload, path, record)

            took = time.perf_counter() - started
            status = record["error"] or summarize(record["output"])
            log(
                f"[{index + 1}/{total_items}] {pending['condition']}: {status} ({took:.1f}s)"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an LLM judge on one bias and save structured JSON results.",
    )
    parser.add_argument("model", help="Ollama model name, for example qwen3")
    parser.add_argument(
        "bias",
        help="Bias name. One of: " + ", ".join(BIASES),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Judge only the first N dataset items",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON file to write. Default: results/<model>__<bias>.json",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--host",
        default=OLLAMA_HOST,
        help=f"Ollama host (default: {OLLAMA_HOST})",
    )
    parser.add_argument(
        "--cr-fraction",
        type=float,
        default=CR_FRACTION,
        help=(
            "Fraction of the full dataset that gets a repeated unperturbed "
            f"judgment for the Consistency Rate (default: {CR_FRACTION})"
        ),
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0 <= args.cr_fraction <= 1:
        parser.error("--cr-fraction must be between 0 and 1")
    args.bias = resolve_bias(args.bias)
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.model,
        arguments.bias,
        arguments.limit,
        arguments.output,
        arguments.temperature,
        arguments.host,
        arguments.cr_fraction,
    )
