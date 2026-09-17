# Historical README (archived 2026-09-17)

Current experiment entrypoints are documented in `scripts/README.md` at the repository root.
The settings below describe older experiments.

# GROOVE

**Group-Relative On-Policy Optimization via Visual Evidence**

This repository implements group-contrastive visual-evidence training. It augments
verl's GRPO advantage with a signed visual-evidence sampled-token OPSD advantage.
The privileged Teacher sees an Analyzer-selected visual prefix; the deployed Student
sees only the original image.

## Implemented objective

For a response token already sampled by the Student, the joint mode builds one
token advantage and sends it through the shared PPO/dual-clip objective:

```text
delta_t = stopgrad(log p_teacher(y_t) - log p_student(y_t))
A_OPSD,t = evidence_mask * delta_t
A_total,t = A_GRPO + 0.01 * A_OPSD,t
L_total = L_PPO/dual-clip(A_total) + 0.001 * L_ref_KL
```

This is the uncentered signed sampled-token reverse-KL estimator: positive gaps
reinforce a sampled token and negative gaps suppress it. There is no sigmoid
gate, trajectory centering, or continuous OCR/DINO reliability coefficient.
The initial evidence-advantage coefficient is `0.01`; optional symmetric gap
clipping is disabled by default. Reference KL uses verl's coefficient `0.001`,
and entropy regularization is disabled.

The terminal GRPO reward is deliberately shaped as

```text
R_answer = 1[extracted answer matches ground truth]
R_format = 1[the response terminates with FINAL: X, for any nonempty X]
R_terminal = 0.9 * R_answer + 0.1 * R_format
```

For the current Vision-OPD-6K multiple-choice data, the answer comparison
accepts a final option letter such as `FINAL: B` or `FINAL: (B)`. The `FINAL: X`
format itself is generic: X is not restricted to A/B/C/D, so the same extraction
protocol can be reused by later non-MCQ data.

- There is no `(1-r_i)` factor.
- GRPO is computed on every rollout.
- If a group has usable visual evidence, signed OPSD credit is computed on every one of its
  eight rollouts, correct and incorrect alike.
- The evidence mask is binary group-level availability, not a confidence or per-rollout reward gate.
- Uniform-reward groups are analyzed too; this preserves the signed OPSD signal
  when a tied group gives GRPO zero advantage. Grounding or Analyzer failures fall
  back to ordinary GRPO.
- The uncentered OPSD term does not change terminal rewards, but it can change a
  trajectory's total token-advantage mass.

## Online training flow

1. Qwen3.5-4B receives the original image and produces eight ordinary-text
   reasoning rollouts. Thinking mode is disabled.
2. The semantic answer evaluator labels each rollout correct or incorrect independently
   from the shaped training reward. An external VLM Analyzer sees the original image,
   question, and two pre-grouped lists containing successful and failed reasoning. It
   never sees numeric rewards, predicted labels, rollout IDs, or the ground-truth answer.
   Degenerate repeated suffixes are removed before reasoning reaches the Analyzer.
3. The Analyzer returns an English inspection instruction and up to three concrete
   English object phrases for the GroundingDINO tool. Comparative visual descriptors
   are allowed, but explicit answer assertions and option letters are rejected.
   Grounding-tool queries remain in English even when the surrounding task text uses
   another language.
4. GroundingDINO-B localizes each phrase independently.  Each object is cropped
   independently with context and enlarged; there is no union crop.
5. During Analyzer tool use, every returned bbox is cropped and sent back as a
   visual feedback image. The Analyzer inspects that preview and revises an English
   query when the crop misses the requested object.
6. The current Student weights act as the no-gradient Teacher on a prefix made
   from the original image, focus text, and the Analyzer-selected crops. Every
   successful tool result receives a candidate ID; the Analyzer selects the best
   one to three candidates after all attempts, without IoU deduplication. Unselected
   boxes remain audit-only. The sampled response is
   unchanged, so Teacher and Student log probabilities align token by token.
7. verl applies the joint loss above.  At inference time the Analyzer, DINO,
   crops, and privileged prefix are removed.

## Repository layout

```text
configs/                 experiment-level design defaults
data/vision_opd/         5,928/313 Vision-OPD-6K train/test manifests
data/vstar/              prepared 191-example V*Bench probe/evaluation data
docs/                    architecture and implementation notes
scripts/                 data and training entrypoints
src/groove/              Analyzer, grounding, evidence, reward, loss, trainer
src/verl/                vendored training runtime used by this project
tests/                   CPU unit tests
TMP/probe_experiments/   archived no-training probes and their reports
TMP/references/          pinned distillation reference material
```

The source JSONL and released bbox/crop files in `../Vision-OPD` are not
rewritten. Data preparation only extracts its archived unboxed original images
in place, then writes the new parquet splits under this repository.

## Reproduce and test

The required verl runtime is vendored under `src/verl`; no sibling repository
or runtime bootstrap step is needed. Prepare the leakage-free Vision-OPD-6K
split and run the CPU tests with the existing Conda environment:

```bash
/home/yzs/miniconda3/envs/vision-opd/bin/python TMP/scripts/prepare_vision_opd.py
PYTHONPATH="$PWD/src" \
  /home/yzs/miniconda3/envs/vision-opd/bin/python -m unittest discover -s tests -v
```

The split uses random state 42 and answer-stratified sampling. It contains 5,928
training examples and 313 held-out examples. Student input uses the unboxed
`original_images` field; the released red-box overlay and Oracle crop are kept
only as audit metadata and are never sent to the Student, Analyzer, or DINO.

## Start training

The default launcher now runs the GRPO-only ablation: OPSD/Teacher/evidence
construction is disabled and the prompt batch is 16. It does not need an
Analyzer endpoint. Start it with:

```bash
./TMP/scripts/run_grpo_ablation.sh
```

`ROLLOUT_N=8` keeps the corresponding PPO mini-batch at 128 sampled rollouts
(`16` prompt groups × `8` rollouts). Use `GROOVE_DRY_RUN=true` to validate the
resolved configuration without starting Ray workers or loading the model. The
existing reference-policy KL regularizer remains enabled; it is separate from
OPSD and is not part of this ablation.

To run the historical joint GRPO+OPSD objective instead, set the switch and
provide the external Analyzer endpoint:

```bash
export OPSD_ENABLED=true
export ANALYZER_BASE_URL='https://your-endpoint.example/v1'
export ANALYZER_API_KEY='...'
export ANALYZER_MODEL='gpt-5.6'
# For the remote visual-tool host used by the training run:
export ANALYZER_USE_VISION_TOOLS=true
export ANALYZER_GROUNDING_URL='http://127.0.0.1:8011'
export ANALYZER_OCR_URL='http://127.0.0.1:8012'
export GROOVE_MAX_CONCURRENCY=8
./TMP/scripts/run_groove.sh
```

The equivalent explicit GRPO-only entrypoint is `TMP/scripts/run_grpo_ablation.sh`;
it forces `OPSD_ENABLED=false` and `TRAIN_BATCH_SIZE=16` even if a shell has
other defaults exported.

The launcher uses the local Qwen3.5-4B weights, the 5,928-example Vision-OPD
training split, its 313-example held-out split, two GPUs, and eight rollouts per
question. In joint mode, Analyzer evidence is built with up to eight concurrent
groups when the three remote endpoints above are set; the remote DINO/OCR
workers remain single-process GPU services. Without those endpoints, probe
configurations fall back to the local GroundingDINO/OCR implementations. The
held-out split is evaluated before training and at the final step. Override any
launcher setting with an environment variable or append a Hydra override.

The two A100-80GB cards use full phase-based GPU time sharing through the
bundled verl hybrid-engine runtime. During rollout, FSDP actor parameters,
optimizer state, and the reference model live on CPU while vLLM owns the GPUs.
Before reference scoring or actor training, vLLM 0.18 enters sleep level 2 and
releases both weights and KV cache. CUDA Graph is disabled so it cannot leave an
unoffloadable GPU allocation; gradient checkpointing and the
Qwen3.5 fused LM head further reduce the training peak. Actor updates use PyTorch
Fused AdamW (`fused=true`, `foreach=false`); its state is still moved to CPU between
updates. The launcher fails early if the installed vLLM cannot provide sleep level 2.

Host RAM is protected separately. This task's cgroup has a hard 220 GiB memory
limit, and Ray's node-wide OOM monitor is pinned to a more conservative effective
threshold. On the current host it fires at 95%, approximately 209 GiB; the Ray
object store is capped at 8 GiB and checked every 100 ms. These defaults are
controlled by `RAY_NODE_MEMORY_CAP_GIB`, `RAY_MEMORY_GUARD_HEADROOM_GIB`,
`RAY_MEMORY_USAGE_THRESHOLD_CEILING`, `RAY_OBJECT_STORE_GIB`, and
`RAY_MEMORY_MONITOR_REFRESH_MS`. See `docs/IMPLEMENTATION.md` for the exact
formula and the distinction between Ray's soft guard and the cgroup hard limit.

Both modes use learning rate `1e-6`, one PPO epoch, clip ratio `0.2`, group
size 8, response length 512, and reference KL coefficient `0.001`. Entropy
regularization is explicitly disabled. The default GRPO-only ablation uses a
16-prompt batch (PPO mini-batch 128 sampled rollouts) and therefore has 370
complete updates over the 5,928-example split (`drop_last=true`); `TOTAL_STEPS=null`
lets verl derive the value from one epoch. Joint mode (`OPSD_ENABLED=true`) additionally uses
signed OPSD advantage coefficient `0.01`, no gap clipping by default, and its
historical 32-prompt batch.
FSDP offload stays enabled because this host has two GPUs rather than an
eight-card reference setup. Checkpoints are written every 500 steps and only
the latest two are retained. For the first end-to-end connectivity check, run
only ten updates:

```bash
TOTAL_STEPS=10 ./TMP/scripts/run_grpo_ablation.sh
```

The GRPO-only run makes no external Analyzer calls. The joint one-epoch run
still makes one Analyzer call per prompt group and should be smoke-tested with
`TOTAL_STEPS=10` before committing to its API cost and wall-clock time.

To validate the complete launcher configuration without creating Ray workers
or loading the model, add `GROOVE_DRY_RUN=true`.

## References

- Vision-OPD paper and official implementation: <https://arxiv.org/abs/2605.18740>,
  <https://github.com/VisionOPD/Vision-OPD>
