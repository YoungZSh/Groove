# Repository guidance for coding agents

## Scope and purpose

This file applies to the repository root and all descendants unless a more
specific `AGENTS.md` exists below a directory.

This project implements GROOVE: normal GRPO plus an uncentered (未中心化),
sampled-token OPSD advantage credit-allocation path derived from privileged
visual evidence. Name the method `GRPO + OPSD` consistently. Do not add public
qualifiers copied from old experiment names, internal identifiers, or papers.

The deployment boundary is important:

- The Student sees only the original image and question.
- The current-policy Teacher scores the exact Student rollout tokens under a
  privileged prefix containing Analyzer-selected visual evidence.
- Analyzer, GroundingDINO, OCR, crops, and Teacher-only text are training-time
  privileges and must never enter Student inference inputs.

## Current implementation sources of truth

The checked-in execution path is authoritative. Read it in this order:

1. `scripts/run_deepeyes_vstar_opsd_2b.sh` and
   `scripts/run_grpo_2b.sh` define the exact current 2B
   experiments. `scripts/run_groove.sh` translates their environment into the
   resolved Hydra/VERL configuration.
2. `src/groove/verl_trainer.py::_postprocess_advantages()` is the integration
   source of truth. It builds online Teacher evidence, computes pre-update
   Teacher log probabilities, constructs OPSD token advantages, combines them
   with the already-computed GRPO advantages, and writes the result back to
   `batch.batch["advantages"]` before the actor update.
3. `src/groove/losses.py::groove_opsd_advantages()` defines the uncentered OPSD
   per-token credit. `combine_grpo_opsd_advantages()` defines the only GRPO +
   OPSD combination used for optimization.
4. `src/groove/objective.py::validate_objective_config()` defines the allowed
   objective configuration: GRPO advantage estimation, one vanilla PPO policy
   loss, reference KL in the loss, and no separate distillation objective.
5. `src/groove/deepeyes_reward.py` defines semantic Judge accuracy, format
   shaping, retry behavior, and repetition handling.
6. `src/groove/analyzer.py`, `src/groove/analyzer_tools.py`, and
   `src/groove/evidence.py` define Analyzer inputs, tool use, crop selection,
   leakage checks, and Teacher prompt construction.
7. `src/verl/` is the active vendored runtime. The tests under `tests/` are the
   executable behavioral contract for all of the above.

Documents under `docs/` explain history and intent but are not implementation
authorities. If prose, a paper, a probe, or an old report differs from the code
path above, follow the current code and update the stale document separately.

The generic README also describes older Vision-OPD/4B configurations. Do not
copy its generic batch size, response length, KL coefficient, checkpoint
cadence, or service topology into a 2B DeepEyes run without checking the 2B
launcher.

Treat files under `Papers/` and `TMP/probe_experiments/` as references and
archived probes, not as executable specifications. Do not derive the current
algorithm from their terminology. Do not modify `TMP/upstream/verl-v0.9.0`;
the active vendored runtime is `src/verl`.

## Repository layout

- `src/groove/`: project objective, Analyzer, evidence, reward, trainer hooks,
  schemas, and diagnostics.
- `src/verl/`: vendored VERL 0.9 runtime with project adaptations.
- `scripts/`: data preparation, launchers, probes, and reporting utilities.
- `remote_tools/`: single-process remote GroundingDINO and PaddleOCR servers.
- `configs/`: shared Hydra defaults.
- `tests/`: CPU unit tests.
- `docs/`: explanatory notes that must be kept aligned with current code.
- `data/`, `outputs/`, `checkpoints/`: local generated state; mostly ignored by
  Git and never safe to delete casually.

## Python and test environment

Use the established environment rather than `/usr/bin/python3`:

```bash
PYTHON_BIN=/home/yzs/miniconda3/envs/vision-opd/bin/python
PYTHONPATH="$PWD/src" "$PYTHON_BIN" -m unittest discover -s tests -v
```

Targeted reward tests:

```bash
PYTHONPATH="$PWD/src" \
  /home/yzs/miniconda3/envs/vision-opd/bin/python \
  -m unittest discover -s tests -p 'test_deepeyes_reward.py' -v
```

Before handing off launcher changes, also run:

```bash
bash -n scripts/run_groove.sh
bash -n scripts/run_grpo_2b.sh
bash -n scripts/run_deepeyes_vstar_opsd_2b.sh
git diff --check
```

Use `GROOVE_DRY_RUN=true` to validate a fully resolved configuration without
starting Ray workers or loading model weights. Always provide a unique
`EXPERIMENT_NAME` when checking a formal launcher.

## Current GRPO + OPSD advantage credit allocation

The current implementation is the uncentered OPSD Advantage version. OPSD is
used for per-token credit allocation at the Advantage level; it is not added as
an independent loss. The executed computation is:

```text
delta_t = stopgrad(log p_teacher(y_t) - log p_student(y_t))
A_OPSD,t = evidence_mask * delta_t
A_total,t = A_GRPO + 0.01 * A_OPSD,t
L_actor = VERL_vanilla_PPO(A_total) + 0.01 * low_var_reference_KL
```

This formula is descriptive shorthand for the current code above; the code is
authoritative. Preserve these properties unless the user explicitly requests
an algorithmic change:

- The Teacher and Student score the identical sampled response tokens.
- Teacher scoring occurs before the actor optimizer update and has no gradient.
- The OPSD token credit is uncentered: do not subtract a trajectory mean or a
  group mean. Do not add a sigmoid gate, `(1-r_i)` factor, or correctness
  multiplier.
- GRPO applies to every rollout. OPSD also applies to correct and incorrect
  rollouts whenever group evidence is available.
- Uniform-reward groups are still analyzed; they can carry OPSD signal even
  when GRPO advantage is zero.
- Analyzer or grounding failure falls back to ordinary GRPO.
- Evidence availability is a binary group-level mask, not a continuous
  DINO/OCR confidence weight.
- `OPSD_ADVANTAGE_COEF=0.01` and no OPSD clipping are the current defaults.

For the current 2B launchers, reference KL uses coefficient `0.01`, training
uses `token-mean`, learning rate `1e-6`, one PPO epoch, and clip ratio `0.2`.

## DeepEyes reward contract

`src/groove/deepeyes_reward.py` deliberately separates semantic correctness
from output formatting:

- The remote Judge returns constrained `0` or `1` semantic `accuracy`.
- Analyzer success/failure grouping uses raw `accuracy`, never shaped `score`.
- Semantic judging may fall back to the complete response when the tag is
  malformed so that format does not silently redefine semantic correctness.
- A valid format has exactly one lowercase, nonempty terminal
  `<answer>...</answer>` pair. Ordinary reasoning before it is allowed; only
  whitespace may follow it. Nested, duplicated, or unbalanced answer tags fail.
- The next 2B runs use `data.response_format=reasoning_answer`: plain reasoning
  followed by the final answer. No think wrapper or reasoning-length gate is
  added. A dataset adapter changes the instruction in memory without rewriting
  historical parquet files. The shared Student/Teacher chat template removes
  Qwen3.5's empty think prefill while keeping `enable_thinking=false`.
- Current next-run shaping is:

```text
format_penalty = 0 if format is valid else -1
score = accuracy + 0.2 * format_penalty
```

Therefore a correct formatted answer scores `1.0`, a correct malformed/bare
answer scores `0.8`, a formatted wrong answer scores `0.0`, and a malformed
wrong answer scores `-0.2`.

The exact contiguous-loop detector is separate. Its defaults require at least
four repetitions covering at least 80 characters. A severely repetitive
trajectory cannot receive positive reward, while raw semantic `accuracy`
remains available for diagnosis and Analyzer grouping.

Do not hard-gate missing tags to `accuracy=0` or skip the Judge without an
explicit decision: that would contaminate Analyzer success/failure groups with
a presentation error. If stricter training becomes necessary, gate `score`
while preserving a separately logged semantic accuracy.

Historical warning: the completed OPSD v4 run used
`FORMAT_REWARD_WEIGHT=0.0`. The tracked launchers now use `0.2` for the next
run. Compare old and new runs by raw `accuracy` as well as shaped `score`.

## Analyzer and evidence boundaries

The Analyzer contract is defined in `src/groove/analyzer.py`.

- Its system prompt is a visual-evidence task, not a self-evolution task. Do
  not reintroduce “self evolution” wording or artifacts.
- Input contains the original image, question, and programmatically separated
  successful/failed reasoning lists.
- It must not receive ground-truth answers, numeric rewards, parsed predicted
  labels, or rollout IDs.
- It must not reassess outcome labels or answer the question.
- Tool descriptions are registered as native tools and injected by the Qwen
  chat template; do not duplicate them in the system prompt.
- Grounding and OCR queries must be short English visual targets.
- The model inspects every returned crop preview and may retry for at most the
  configured tool rounds.
- Every usable crop has a candidate ID. The Analyzer makes the final selection
  of one to three candidates. There is no IoU deduplication or implicit
  latest-crop preference.
- One crop is sufficient for a single target. Multiple images are allowed for
  comparison, counting, or spatial relationships.
- `visible_focus_instruction` must be answer-neutral and must not leak OCR text,
  answer assertions, option letters, rewards, or rollout outcomes.

Keep crop images and exact tool traces private to the Teacher/audit path.
Unselected crops remain audit-only.

## Current 2B experiment settings

The comparable GRPO and OPSD runs use:

- Model: `/root/siton-tmp/yzs/ckpts/Qwen3.5-2B`
- Seed: `20260904`
- Two GPUs: `CUDA_VISIBLE_DEVICES=0,1`
- Prompt batch: 16 groups
- Rollouts per group: 8
- Maximum response length: 1024 tokens
- Training sampling: temperature `1.0`
- Validation sampling: temperature `0`, no sampling
- Validation and checkpoint cadence: every 10 steps
- Retained actor checkpoints: 2
- Thinking mode: disabled
- W&B mode during training: offline

Never assume identical seeds alone imply a fair comparison. Confirm the model,
parquet row order, shuffle settings, batch size, rollout count, Judge protocol,
reward weights, and validation temperature.

Completed reference runs:

- Pure GRPO W&B ID `iam35gwn`: final Validation `0.6636`, best `0.6773`.
- OPSD v4 W&B ID `5zmaf7ds`: final Validation `0.7364`, best `0.7455`.
- Both contain 123 online history points after final local-log backfill.

## Service topology

The formal OPSD launcher expects three local HTTP endpoints that front the
deployed services:

- `127.0.0.1:8002/v1`: Qwen3.8-27B Analyzer and binary Judge.
- `127.0.0.1:8011`: single-worker GroundingDINO service.
- `127.0.0.1:8012`: single-worker PaddleOCR service.

Check them read-only before a run:

```bash
curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8011/ >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8012/ >/dev/null
```

Analyzer group orchestration may use concurrency 16. DINO and OCR must remain
one worker each unless the user explicitly approves a remote deployment
change. Editing `remote_tools/*.py` locally does not update a remote machine or
restart a running service.

Do not print API keys, `.netrc`, environment secrets, or request authorization
headers. Placeholder values such as `remote-qwen38` are not user credentials.

## Running and monitoring training

- Do not start, stop, resume, or kill a formal training job unless the user asks.
- Prefer a fresh run over resuming an interrupted or completed run unless the
  user explicitly requests resume behavior.
- Set a new `EXPERIMENT_NAME` for every formal run. The launcher currently has
  historical names as defaults, and VERL uses auto-resume; invoking it without
  a new name can collide with an existing checkpoint directory.
- Do not delete or overwrite checkpoints, rollouts, evidence, logs, or W&B
  offline files.
- Avoid changing reward or training source while a run is active. If a change
  is intended for the next run, state that explicitly and verify the active
  run's resolved W&B config before comparing results.
- Inspect GPU/RAM and service health before launch. Use tmux or the established
  supervisor for long jobs and preserve the full log.

Generated state is organized by `EXPERIMENT_NAME`:

- `outputs/logs/`
- `outputs/rollouts/`
- `outputs/evidence/`
- `outputs/opsd-token-dumps/`
- `outputs/wandb/wandb/`
- `checkpoints/`

Monitor at least:

- rollout `reward/answer_reward_mean` and Validation reward/accuracy;
- strict `has_answer_tag` / `format_valid` rate;
- severe-repetition rate and response length;
- `actor/kl_loss`, entropy, and gradient norm;
- evidence ready/error/fallback fractions and Analyzer wall time;
- OPSD-to-GRPO advantage RMS ratio, credit-direction agreement, and clipping
  fraction;
- step wall time and checkpoint/validation overhead.

Training reward is noisy. Prefer fixed Validation points and multi-step trends
over conclusions from one rollout batch. When reward shaping differs between
runs, compare raw semantic accuracy separately.

## W&B handling

Formal launchers log offline under `outputs/wandb`. Upload only when the user
requests it. Locate the exact offline directory by its logged
`trainer.experiment_name`; do not sync every historical probe by accident.

After sync, verify through the W&B API that:

- the run name and ID match the intended experiment;
- state is `finished`;
- history reaches the local final `training/global_step`;
- final and best Validation values match the local log.

The SDK can fail to flush the last one or two steps at process exit. If the
online history is short, backfill only the missing numeric records from the
corresponding local log, record the source-log hash, and verify again. Never
invent or interpolate missing training metrics.

The current next-run code snapshot is W&B artifact
`mmcot-opsd-source-format-penalty:v0`, associated with code-snapshot Run ID
`oqsxbci5`. It matches Git commit `a6c9f6c`; always resolve the current revision
with `git rev-parse HEAD` rather than assuming that snapshot is still latest.

## Git and editing discipline

- Preserve unrelated user changes and inspect `git status` before editing.
- Use focused patches and keep vendored VERL modifications minimal.
- Add or update tests for every reward, prompt-schema, advantage, or launcher
  behavior change.
- Update the relevant architecture document when an invariant changes.
- Run the targeted tests, shell syntax checks, and `git diff --check` before
  committing.
- Group commits by coherent behavior. Do not commit generated data, model
  weights, logs, rollouts, evidence, W&B directories, or checkpoints.
- Do not push Git commits or mutate remote services unless explicitly asked.
- Report the commit hash, tests run, and whether the worktree is clean.

When reporting status to the user, default to concise Chinese and distinguish
observed facts from interpretation.
