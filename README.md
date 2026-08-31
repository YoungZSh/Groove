# Multimodal CoT Visual-SEED

This repository is the first trainable version of the group-contrastive visual
self-evolution idea discussed in this task.  It combines a normal verl GRPO
update with SEED-style sampled-token OPD.  The privileged Teacher sees an
Analyzer-selected visual prefix; the deployed Student sees only the original
image.

## Implemented objective

For a response token already sampled by the Student,

```text
delta_t = stopgrad(log p_teacher(y_t) - log p_student(y_t))
gate_t  = sigmoid(beta * delta_t)
L_OPD   = mean(gate_t * (stopgrad(log p_teacher(y_t)) - log p_student(y_t)))
L_total = L_GRPO/PPO-clip + 0.01 * L_OPD + 0.001 * L_ref_KL
```

The OPD values follow the SEED recipe: `lambda=0.01`, `beta=5.0`. Reference
KL uses verl's default coefficient `0.001`; entropy regularization is disabled.

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
- If a group has usable visual evidence, OPD is computed on every one of its
  eight rollouts, correct and incorrect alike.
- The evidence mask is group-level availability, not a per-rollout reward gate.
- Uniform-reward groups are analyzed too; this preserves SEED's useful OPD signal
  when a tied group gives GRPO zero advantage. Grounding or Analyzer failures fall
  back to ordinary GRPO.

## Online training flow

1. Qwen3.5-4B receives the original image and produces eight ordinary-text
   reasoning rollouts. Thinking mode is disabled.
2. The rule reward forms a GRPO group. An external VLM Analyzer sees the original
   image plus all eight trajectories, predictions, and terminal rewards. A reward
   above `0.5` denotes an answer-correct rollout; `0.1` versus `0.0` only records
   whether an incorrect rollout used the `FINAL: X` protocol. It contrasts correct
   and incorrect traces when both exist, and otherwise compares reasoning variations
   to recover shared or missing visual evidence. It never sees the ground-truth label.
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
   from the original image, focus text, and the final confirmed crops.  Superseded
   retry boxes remain audit-only. The sampled response is
   unchanged, so Teacher and Student log probabilities align token by token.
7. verl applies the joint loss above.  At inference time the Analyzer, DINO,
   crops, and privileged prefix are removed.

## Repository layout

```text
configs/                 experiment-level design defaults
data/vision_opd/         5,928/313 Vision-OPD-6K train/test manifests
data/vstar/              prepared 191-example V*Bench probe/evaluation data
docs/                    architecture and implementation notes
patches/                 reproducible patch against Vision-OPD's verl fork
scripts/                 data, runtime bootstrap, and training entrypoints
src/mmcot_opsd/          Analyzer, grounding, evidence, reward, loss, trainer
tests/                   CPU unit tests
TMP/probe_experiments/   archived no-training probes and their reports
TMP/references/          pinned clean SEED and Vision-OPD references
TMP/runtime/Vision-OPD/  patched clean training runtime
```

The source JSONL and released bbox/crop files in `../Vision-OPD` are not
rewritten. Data preparation only extracts its archived unboxed original images
in place, then writes the new parquet splits under this repository.

## Reproduce and test

The prepared runtime is already present. To reconstruct it at the pinned
Vision-OPD commit in a new workspace:

```bash
./scripts/bootstrap_runtime.sh
```

Prepare the leakage-free Vision-OPD-6K split and run the CPU tests:

```bash
/home/yzs/miniconda3/envs/vision-opd/bin/python scripts/prepare_vision_opd.py
PYTHONPATH="$PWD/src:$PWD/TMP/runtime/Vision-OPD" \
  /home/yzs/miniconda3/envs/vision-opd/bin/python -m unittest discover -s tests -v
```

The split uses seed 42 and answer-stratified sampling. It contains 5,928
training examples and 313 held-out examples. Student input uses the unboxed
`original_images` field; the released red-box overlay and Oracle crop are kept
only as audit metadata and are never sent to the Student, Analyzer, or DINO.

## Start training

Set the external Analyzer endpoint when its credentials are available:

```bash
export ANALYZER_BASE_URL='https://your-endpoint.example/v1'
export ANALYZER_API_KEY='...'
export ANALYZER_MODEL='gpt-5.6'
./scripts/run_visual_seed.sh
```

The launcher uses the local Qwen3.5-4B weights, the 5,928-example Vision-OPD
training split, its 313-example held-out split, two GPUs, eight rollouts per
question, and the local Hugging Face GroundingDINO-B cache. DINO defaults to CPU
to avoid competing with the two training GPUs. The held-out split is evaluated
before training and at the final step. Override any launcher setting with an
environment variable or append a Hydra override.

The two A100-80GB cards use full phase-based GPU time sharing, following the
local Vision-OPD hybrid-engine setup. During rollout, FSDP actor parameters,
optimizer state, and the reference model live on CPU while vLLM owns the GPUs.
Before reference scoring or actor training, vLLM 0.18 enters sleep level 2 and
releases both weights and KV cache. CUDA Graph is disabled so it cannot leave an
unoffloadable GPU allocation; gradient checkpointing and the
Qwen3.5 fused LM head further reduce the training peak. Actor updates use PyTorch
Fused AdamW (`fused=true`, `foreach=false`); its state is still moved to CPU between
updates. The launcher fails early if the runtime cannot provide sleep level 2.

Host RAM is protected separately. This task's cgroup has a hard 220 GiB memory
limit, and Ray's node-wide OOM monitor is pinned to a more conservative effective
threshold. On the current host it fires at 95%, approximately 209 GiB; the Ray
object store is capped at 8 GiB and checked every 100 ms. These defaults are
controlled by `RAY_NODE_MEMORY_CAP_GIB`, `RAY_MEMORY_GUARD_HEADROOM_GIB`,
`RAY_MEMORY_USAGE_THRESHOLD_CEILING`, `RAY_OBJECT_STORE_GIB`, and
`RAY_MEMORY_MONITOR_REFRESH_MS`. See `docs/IMPLEMENTATION.md` for the exact
formula and the distinction between Ray's soft guard and the cgroup hard limit.

The optimization defaults follow the public SEED recipe where it transfers
cleanly: learning rate `1e-6`, one PPO epoch, clip ratio `0.2`, group size 8,
response length 512, OPD coefficient `0.01`, gate beta `5`, and reference KL
coefficient `0.001`. Entropy regularization is explicitly disabled. The prompt
batch remains 2 and FSDP offload stays enabled because this host has two GPUs
rather than SEED's eight A800-80GB setup. One full split pass is 2,964 updates;
`TOTAL_STEPS=null` lets verl derive that value from one epoch. Checkpoints are
written every 500 steps and only the latest two are retained. For the first
end-to-end connectivity check, run only ten updates:

```bash
TOTAL_STEPS=10 ./scripts/run_visual_seed.sh
```

The full one-epoch run makes 2,964 external Analyzer calls and 47,424 Student
rollouts, so the ten-step connectivity run should be completed before committing
to its API cost and wall-clock time.

To validate the complete launcher configuration without creating Ray workers
or loading the model, add `VISUAL_SEED_DRY_RUN=true`.

## References

- SEED paper and official implementation: <https://arxiv.org/abs/2607.14777>,
  <https://github.com/jinyangwu/SEED>
- Vision-OPD paper and official implementation: <https://arxiv.org/abs/2605.18740>,
  <https://github.com/VisionOPD/Vision-OPD>
