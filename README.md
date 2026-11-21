# Do small groups of experts control refusal in mixture-of-experts (MoE) language models?

This repository turns off chosen groups of experts at inference time in Sarvam-30B, GPT-OSS-20B and Qwen3.5-35B-A3B. For each
group it measures how often the model then engages with harmful requests and how often it refuses harmless
ones. Each group is compared against random expert groups matched on how much the router uses them.

## Method

1. **Selection rule (gradient attribution).** The routing weights are treated as leaves with routing frozen.
   The score is the margin between the log-probability of the model's own refusal openings and that of its
   compliance openings. An expert's score is the mean of (d margin / d weight) × weight, summed over tokens and
   averaged over refused discovery answers. The group is the top N experts, with at most 4 per layer.
2. **Removal.** A vLLM router hook edits the selected routing weights. The operators are `suppress` (weight ×
   alpha, alpha 0 = zero), `exclude` (block the expert so the router picks the next one), `mass_restore`
   (zero it and rescale the rest) and `steer` (SteerMoE-style deactivation and activation).
   Before any run, a hook audit checks that the no-op hook is exact, that an inactive expert changes nothing,
   that alpha = 1 is exact and that removal changes the layer's output.
3. **Matched controls.** Each selected expert gets 5 random substitutes from its own layer. Each substitute is
   drawn from the 8 experts nearest in routing frequency and mass on the discovery prompts.
4. **Judge.** gpt-5-nano (OpenAI Batch API) labels every answer. Harmful prompts are scored with the HEx-PHI
   1–5 rubric; *engaged* means a score of 2 or more. Harmless prompts get the XSTest classes; *any refusal*
   means full or partial refusal.

Decoding is greedy everywhere. Prompts run in fixed batches, so a baseline batch and an intervention batch
contain the same prompts in the same order.

## Repository layout

```
configs/
  sarvam/experiments.json        engine flags and seed for each named Sarvam experiment
  sarvam/pilot_seed/             routing activity and calibration used for matched controls
  sarvam/groups_*.json           groups of the pilot, the 32/64-expert extension and SteerMoE
  gpt_oss/, qwen35/              selected groups and judged discovery labels
data/
  pairs.json                     30 discovery + 48 pilot pairs (HEx-PHI harmful request + harmless rewrite)
  fresh_test_bank.jsonl          299 held-out pairs for the pre-registered fresh test
  overrefusal_bank.jsonl         128 OR-Bench-Hard harmless + 128 held-out harmful prompts
  gsm8k/test.jsonl               GSM8K test set (ability check)
  sarvam/historical_ranking.json attribution ranking from the earlier study
  steermoe/                      BeaverTails detection set for SteerMoE
requirements/                    one file per environment (see below)
scripts/                         Modal launchers: modal_sarvam.py, modal_gpt_oss.py, modal_qwen35.py
src/refusal_moe/
  sarvam/                        engine, experiment driver, groups, seeds, SteerMoE, vLLM plugin
  gpt_oss/                       runner, baseline, selection, pilot, vLLM plugin
  qwen35/                        runner, baseline, selection, pilot, vLLM plugin
  judge/                         gpt-5-nano batch judge and the two rubrics
  download.py, matching.py, io.py, paths.py
```

## Environments

| Env | File | Python / CUDA | Key pins | Used by |
|---|---|---|---|---|
| `vllm015` | `requirements/vllm015.txt` (full lock) | 3.12, CUDA 12.8 wheels | vLLM 0.15.0, torch 2.9.1, transformers 4.57.3 | `refusal_moe.sarvam.*`, `refusal_moe.gpt_oss.*` |
| `vllm033` | `requirements/vllm033.txt` (full lock) | 3.12, CUDA 13.0 wheels | vLLM 0.33.0, torch 2.14.0, transformers 5.20.2 | `refusal_moe.qwen35.*` |
| `client` | `requirements/client.txt` | 3.11–3.12, no GPU | modal 1.6.4, openai 3.46.0 | `scripts/*`, `refusal_moe.judge.*`, `refusal_moe.download` |

The two GPU environments conflict: vLLM 0.15.0 cannot load Qwen3.5, and vLLM 0.33.0 needs transformers 5.
For `refusal_moe.download` alone, install `huggingface-hub` in the client environment.

Hardware used:

| Model | GPU | Notes |
|---|---|---|
| Sarvam-30B | one 80 GB GPU (A100) | BF16 weights, about 60 GiB |
| GPT-OSS-20B | one A100 80 GB | the hook needs the Marlin MXFP4 backend |
| Qwen3.5-35B-A3B | one H200 | BF16 weights, about 67 GiB |

## Setup

```bash
git clone <this repo> && cd refusal-expert-moe
cp .env.example .env    # fill in HF_TOKEN and OPENAI_API_KEY
```

Each environment, with uv (pip works the same way):

```bash
uv venv .venv-vllm015 --python 3.12
VIRTUAL_ENV=.venv-vllm015 uv pip install -r requirements/vllm015.txt
VIRTUAL_ENV=.venv-vllm015 uv pip install --no-deps -e .
```

```bash
uv venv .venv-vllm033 --python 3.12
VIRTUAL_ENV=.venv-vllm033 uv pip install -r requirements/vllm033.txt
VIRTUAL_ENV=.venv-vllm033 uv pip install --no-deps -e .
```

```bash
uv venv .venv-client --python 3.12
VIRTUAL_ENV=.venv-client uv pip install -r requirements/client.txt
VIRTUAL_ENV=.venv-client uv pip install --no-deps -e .
```

The editable install registers the Sarvam model class with vLLM, through the `vllm.general_plugins` entry
point. Run every command from the repository root; paths resolve against it, or against `REFUSAL_MOE_ROOT`
if that is set.

## Models and data

```bash
set -a && . ./.env && set +a
python -m refusal_moe.download sarvam-30b --to models/sarvam-30b
python -m refusal_moe.download gpt-oss-20b --to models/gpt-oss-20b
python -m refusal_moe.download qwen35-35b-a3b --to models/qwen35-35b-a3b
```

Revisions are pinned in `src/refusal_moe/download.py`. Each download writes
`models/<name>/download_receipt.json` with a sha256 for every file, and the runners refuse a checkpoint
whose revision does not match.

All prompt data ships in `data/`. Sources: HEx-PHI, JailbreakBench, OR-Bench-Hard-1K, GSM8K (MIT) and
BeaverTails (CC BY-NC 4.0). Check each dataset's terms before redistributing.

## Local runs

### Sarvam-30B (`vllm015`)

```bash
export SARVAM_ROUTER_LOGITS_FP32=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Each named experiment in `configs/sarvam/experiments.json` is seeded with its groups if needed, then run.
Rerunning the same command resumes an interrupted run.

```bash
python -m refusal_moe.sarvam.launch pilot --run-dir runs/sarvam_pilot
python -m refusal_moe.sarvam.launch size-extension --run-dir runs/sarvam_size_extension
python -m refusal_moe.sarvam.launch steermoe --run-dir runs/sarvam_steermoe
python -m refusal_moe.sarvam.launch l5e65 --run-dir runs/sarvam_l5e65
python -m refusal_moe.sarvam.launch fresh-test --run-dir runs/sarvam_fresh_test
python -m refusal_moe.sarvam.launch removal-methods --run-dir runs/sarvam_removal_methods
python -m refusal_moe.sarvam.launch overrefusal --run-dir runs/sarvam_overrefusal
```

| Experiment | Configurations | Prompts per configuration |
|---|---|---|
| `pilot` | baseline; gradient, routing-contrast and complementary groups of 1–16 experts; single experts; operators; 5 controls each | 48 pilot pairs |
| `size-extension` | the same rankings at 32 and 64 experts | 48 pilot pairs |
| `steermoe` | SteerMoE detection, then its top 16/32/64 (off) and top 5 (on) | 48 pilot pairs + 48 GSM8K |
| `l5e65` | layer 5, expert 65 alone, and the gradient groups without it | 48 pilot pairs + 48 GSM8K |
| `fresh-test` | the 32-expert set (two control sets) and SteerMoE's top 32 under two operators (25 in all) | 299 fresh pairs + 48 GSM8K |
| `removal-methods` | the 32-expert groups under 6 removal methods (73 in all) | 48 pilot pairs + 48 GSM8K |
| `overrefusal` | the 32-expert set at zero and half weight; baseline run twice (14 in all) | 128 + 128 prompts |

Two GPUs with a pipeline split:
`VLLM_PP_LAYER_PARTITION=14,5 python -m refusal_moe.sarvam.launch pilot --run-dir runs/sarvam_pilot --placement two_gpu`.

The engine can also be called directly: `python -m refusal_moe.sarvam.engine --help`. Seeds can be written
without running them: `python -m refusal_moe.sarvam.seeds fresh-test --to runs/sarvam_fresh_test`.

### GPT-OSS-20B (`vllm015`)

```bash
export VLLM_MXFP4_USE_MARLIN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m refusal_moe.gpt_oss.baseline --model models/gpt-oss-20b --run-dir runs/gptoss_baseline
python -m refusal_moe.gpt_oss.selection generate --model models/gpt-oss-20b --run-dir runs/gptoss_discovery
python -m refusal_moe.gpt_oss.selection gradient --model models/gpt-oss-20b --run-dir runs/gptoss_selection --generations runs/gptoss_discovery --labels configs/gpt_oss/discovery_labels.jsonl
python -m refusal_moe.gpt_oss.pilot --model models/gpt-oss-20b --run-dir runs/gptoss_pilot_n10 --size 10 --selection configs/gpt_oss/selection_group_n10.json
python -m refusal_moe.gpt_oss.pilot --model models/gpt-oss-20b --run-dir runs/gptoss_pilot_n32 --size 32 --selection configs/gpt_oss/selection_group_n32.json
```

### Qwen3.5-35B-A3B (`vllm033`)

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m refusal_moe.qwen35.baseline --model models/qwen35-35b-a3b --run-dir runs/qwen35_baseline
python -m refusal_moe.qwen35.selection generate --model models/qwen35-35b-a3b --run-dir runs/qwen35_discovery
python -m refusal_moe.qwen35.selection gradient --model models/qwen35-35b-a3b --run-dir runs/qwen35_selection --generations runs/qwen35_discovery --labels configs/qwen35/discovery_labels.jsonl
python -m refusal_moe.qwen35.pilot --model models/qwen35-35b-a3b --run-dir runs/qwen35_pilot --size 143 --selection configs/qwen35/selection_group_n143.json
```

The `--labels` files in `configs/` are the judged discovery answers used for the reported groups. To make
fresh labels, judge the discovery run with `--phases discovery` (see below), then pass
`runs/<discovery run>_judge/judge_labels.jsonl` instead.

### Judge (`client`)

```bash
set -a && . ./.env && set +a
python -m refusal_moe.judge.batch build --run runs/sarvam_fresh_test
python -m refusal_moe.judge.batch drive --run runs/sarvam_fresh_test --poll 120
```

The judge writes `runs/<run>_judge/judge_labels.jsonl`. `--phases` selects what to judge: the default
`pilot` covers pilot answers, and `discovery` covers selection inputs. GPT-OSS and Qwen3.5 runs are read
directly, and only their final answers are judged.

## Modal runs

Use your own Modal account and Hugging Face token:

```bash
modal token new
modal profile activate <YOUR_PROFILE_NAME>
modal secret create huggingface HF_TOKEN=<YOUR_HF_TOKEN>
```

`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment also work. Set `HF_SECRET_NAME` to use a
secret with another name. Weights go to the volume `refusal-moe-weights`, and runs go to `refusal-moe-runs`;
each run is mirrored there every 5 minutes.

```bash
modal run scripts/modal_sarvam.py::download
modal run --detach scripts/modal_sarvam.py::launch --task fresh-test --name sarvam_fresh_test
modal run scripts/modal_sarvam.py::progress --name sarvam_fresh_test
modal run scripts/modal_sarvam.py::fetch --name sarvam_fresh_test
```

Sarvam tasks are the experiment names above. Relaunching a Sarvam name resumes it.

```bash
modal run scripts/modal_gpt_oss.py::download
modal run --detach scripts/modal_gpt_oss.py::launch --task discovery --name gptoss_discovery
modal run --detach scripts/modal_gpt_oss.py::launch --task selection --name gptoss_selection --source gptoss_discovery
modal run --detach scripts/modal_gpt_oss.py::launch --task pilot-n10 --name gptoss_pilot_n10
modal run scripts/modal_gpt_oss.py::fetch --name gptoss_pilot_n10
```

GPT-OSS tasks: `baseline`, `discovery`, `selection`, `pilot-n10`, `pilot-n32`.

```bash
modal run scripts/modal_qwen35.py::download
modal run --detach scripts/modal_qwen35.py::launch --task pilot --name qwen35_pilot
modal run scripts/modal_qwen35.py::fetch --name qwen35_pilot
```

Qwen3.5 tasks: `baseline`, `discovery`, `selection`, `pilot`.

`fetch` copies the run to `runs/<name>`, ready for the judge.

## Expected outputs

| Step | Files in the run directory |
|---|---|
| Any Sarvam run | `audit.json`, `engine_settings.json`, `model_representation.json`, `packages.json`, `hardware.json`, `identity.json`, `protocol.json`, `sessions.jsonl` |
| Sarvam experiment | `groups.json`, `schedule.json`, `generations.jsonl`, `activity.jsonl`, `calibration_baseline.json`, `calibration.jsonl`; SteerMoE adds `steermoe_detect.jsonl`, `steermoe_selection.json` |
| baseline | `audit.json`, `engine_settings.json`, `model_representation*.json`, `generations.jsonl` |
| selection | `generations.jsonl` (generate); `attribution.npz`, `openings.json`, `hf_load.json`, `selection.json` (gradient; Qwen3.5 also `selection_group_n143.json`) |
| pilot | `audit.json`, `activity.jsonl`, `groups.json`, `generations.jsonl` |
| judge | `<run>_judge/judge_labels.jsonl`, `judge_config.json`, `merge_summary.json` |


## License

Apache-2.0 (see `LICENSE`). Datasets keep their own licenses.
