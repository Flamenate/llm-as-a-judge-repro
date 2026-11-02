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

Writes CSV tables and the presentation's French charts to `analysis/`. Where the paper reports the same metric, its six judges are drawn as a box plot and our models as points.

## Presentation

The slides are read from `presentation/presentation_fr.md` each time the page loads. Serve the repository root, then open the audience view:

```bash
python -m http.server 8000
```

`http://localhost:8000/presentation/web/`

Press `P` (or the screen button, bottom right) to open the presenter view. It shows the speech, the current slide, a timer and the slide number, and stays in sync with the audience window: either window can drive the other. Libraries and fonts are in `presentation/web/vendor/`, so no network is needed.

| Key | Action |
| --- | --- |
| `→` `↓` `Space` `Enter` `PageDown` | Next slide |
| `←` `↑` `Backspace` `PageUp` | Previous slide |
| `Home` / `End` | First / last slide |
| number then `Enter` | Go to that slide |
| `F` | Full screen (audience view) |
| `P` | Open the presenter view (audience view) |
| `B` | Black screen |
| `T` | Pause or resume the timer (presenter view) |
| `+` / `-` | Speech text size (presenter view) |

With the mouse, click the slide to go forward (the left quarter goes back), use the buttons that appear when the mouse moves, or click a segment of the progress bar at the top.
