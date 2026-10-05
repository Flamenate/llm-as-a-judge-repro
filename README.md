# LLM-as-a-judge bias runner

Reproduces the pairwise and scoring judgments from Ye et al., [Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge](https://arxiv.org/abs/2410.02736). Prompts come from `promptTemplate.py`. A local or remote [Ollama](https://ollama.com) server runs the judge model.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
git clone https://github.com/Y0oMu/LLM-Judge-Bias-Dataset
```

The dataset clone must sit at `LLM-Judge-Bias-Dataset/` next to `judge.py`.

## Run a bias

```bash
python judge.py qwen3:4b position --host 127.0.0.1:11434
```

`model` is the Ollama model name. `bias` is one of:

`position`, `verbosity`, `compassion`, `bandwagon`, `distraction`, `fallacy`, `authority`, `sentiment`, `diversity`, `cot`, `self-enhancement`, `refinement`

Underscores are accepted (`chain_of_thought` is `cot`). Aliases such as `fallacy-oversight` and `refinement-aware` also work.

| Flag | Meaning |
| --- | --- |
| `--host` | Ollama address. Default is the host set in `judge.py`. |
| `--limit N` | Judge only the first N items. |
| `--temperature` | Default `0.7`. |
| `--output` | JSON path. Default is `results/<model>__<bias>.json`. |

Results are saved after every call. Ctrl+C stops the run and keeps what was written; run the same command again to continue. Calls that already have a verdict are skipped. A missing optional field (for example a citation on an authority item) skips that condition and the run continues.

## Summarize results

```bash
python analyze.py
python analyze.py --models qwen3:4b --biases position authority
```

Writes CSV tables and PNG charts to `analysis/`.
