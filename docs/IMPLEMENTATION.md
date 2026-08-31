# Implementation decisions

## What comes from each reference

From SEED, this project adopts the sampled-token teacher/student log-probability
gap, the detached sigmoid token gate, and the additive GRPO + OPD objective.
SEED's skill generation machinery is not copied: here the retained skill is a
multimodal Crop/Zoom prefix generated from group-level rollout contrasts.

From Vision-OPD, this project reuses the Qwen3.5/verl integration and its tested
ability to score one fixed response under two prompts with different multimodal
inputs.  Its original full-logit VOPD KL/JSD path remains intact.  The new
`visual_seed` mode avoids full-vocabulary logits and uses only sampled-token
log probabilities.

## Information boundaries

The Analyzer input contains:

- original image and multiple-choice question;
- eight sampled reasoning traces;
- each trace's parsed prediction and terminal reward.

The terminal reward has two independent components:

```text
answer_reward = 1[parsed answer matches ground truth]
format_reward = 1[response ends in FINAL: X for any nonempty X]
score = 0.9 * answer_reward + 0.1 * format_reward
```

The answer component remains useful even when the format is missing: a correct
multiple-choice answer in ordinary response text receives `0.9`, and the missing
`0.1` provides a clean incentive to emit the final answer anchor. For current
Vision-OPD-6K labels, matching is letter-aware (`B`, `(B)`, or `FINAL: (B)` all
match the label `B`); the `FINAL: X` grammar itself imposes no A/B/C/D restriction.
Consequently, `reward > 0.5` is the correct/incorrect boundary exposed to the
Analyzer and group logic, while `format_reward` is logged separately.

It does not contain `reward_model.ground_truth` or `extra_info.answer`.
Nevertheless, a successful rollout plus its binary reward makes the answer
inferable to the Analyzer; that hindsight is intentional, while direct answer
leakage into Teacher-visible text is not. The Analyzer's visible instruction is
required to describe an inspection action, not an option or conclusion. Comparative
visual descriptors such as candidate colors or shapes are allowed when they tell
the Teacher what to inspect; explicit answer assertions and option letters are
still rejected. Analyzer messages and GroundingDINO queries use English; the latter
is an explicit tool contract because the deployed DINO checkpoint is English-oriented.
After each visual-tool call, the exact returned crop is sent back to the Analyzer as
an image message so it can verify or revise the query before finalizing the focus
program. Only the latest successful tool round selected for the final focus is
promoted to the Teacher; superseded retry boxes remain audit-only. Exact grounding
phrases stay in the private execution record and are not Student targets.

## Mask semantics

`self_distillation_mask[i] = 1` means that the group containing rollout `i`
has a valid visual Teacher prefix.  Every rollout in that group receives the
same value. It never means that rollout `i` was wrong.  This makes the initial
objective:

```text
all rollout tokens       -> GRPO
tokens in evidence group -> GRPO + lambda * SEED-OPD
tokens in fallback group -> GRPO only
```

The default eligibility rule is now simply a successful Analyzer/DINO execution.
Mixed, all-correct, and all-wrong groups can all receive OPD. This matches SEED's
default `failed_only=False` behavior and, importantly, keeps auxiliary learning
alive on tied-reward groups where the GRPO advantage is zero. The compatibility
switch `VISUAL_SEED_MIXED_GROUPS_ONLY=true` restores the earlier mixed-only probe.

## SEED code alignment

The official SEED repository is open source. The following settings are matched
to its public paper and visual-task launcher:

| Detail | SEED | This project |
| --- | --- | --- |
| rollout group | 8 | 8 |
| sampled-token gate | `sigmoid(5 * delta)` | same |
| OPD coefficient | 0.01 | 0.01 |
| actor learning rate | 1e-6 | 1e-6 |
| PPO clip | 0.2 | 0.2, explicitly overriding Vision-OPD's user preset |
| PPO epochs | 1 | 1 |
| entropy coefficient | 0.001 in the SEED code defaults | **0.0 (disabled)** |
| reference KL | low-var KL, coefficient 0.01 | **0.001 (verl default)** |
| GRPO advantage | group standard-deviation normalization | same |
| Analyzer temperature | 0.0 | 0.0 |
| Analyzer completion cap | 2048 for the public visual launcher | 2048 |
| total steps | 150 in the paper | one 5,928-example epoch (2,964 updates); 150 is the pilot override |

The host-dependent values intentionally remain smaller: prompt batch 2 and PPO
mini-batch 16 (one complete 2x8 rollout batch), micro-batch 1, and parameter/
optimizer/reference offload. SEED reports prompt batch 16, PPO mini-batch 128,
micro-batch 8, and no actor offload on eight A800 80GB GPUs. The fine-grained VQA
data also keeps a 4096-token prompt budget rather than SEED EZPoints' 1024 because
its image/question format and multiple visual prefixes differ.

One algorithmic difference is deliberate. Public SEED analyzes each trajectory
separately and creates a per-trajectory text skill. Here the external Analyzer sees
all eight trajectories jointly, performs the requested success/failure contrast,
and produces one shared multimodal Crop/Zoom skill for the group. The loss and
detached teacher/gate semantics stay SEED-compatible; only skill construction is
group-contrastive.

The regularization choice is also task-specific: the training loss contains GRPO,
sampled-token OPD, and verl's reference-policy KL only. Entropy regularization is
disabled, and the KL coefficient uses verl's `0.001` default instead of SEED's
`0.01`, because this is a short single-turn VQA task rather than a
long-horizon exploration environment.

A second deliberate difference is Analyzer evolution. Paper-style SEED first uses
an external model to annotate 1,440 trajectories (180 tasks x 8 rollouts), trains
the policy for three SFT epochs to acquire trajectory-analysis ability, and then
uses the latest policy snapshot as both actor and `policy_vllm` Analyzer during RL.
Version one here skips that Stage-1 SFT and keeps the requested external GPT-5.6
Analyzer fixed. The Student/privileged-Teacher model still self-evolves on-policy,
but the visual-focus generator does not yet evolve. This is a clean ablation axis:
once the external-Analyzer experiment works, its accepted focus programs can form
the SFT data for a later synchronized Qwen Analyzer.

## Self-evolution behavior

`teacher_model_source=current` causes both passes to use the currently training
model.  The difference is privileged visual context, not a larger frozen
Teacher.  As the Student improves, the next on-policy group and the next
privileged Teacher distribution both change.  The external Analyzer and DINO
remain fixed in version one, which removes an additional drifting learned
policy from the loop.

## Runtime patch

`patches/vision_opd_visual_seed.patch` applies cleanly to Vision-OPD commit
`c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471`. It adds:

- `compute_seed_opd_loss` in verl core algorithms;
- the `visual_seed` actor branch;
- `opd_loss_coef` and `opd_gate_beta` configuration;
- multimodal self-distillation activation for the new mode;
- a `visual_seed.yaml` Hydra preset.

Project-side `VisualSeedRayPPOTrainer` constructs the online evidence columns
and then delegates multimodal tokenization/alignment to Vision-OPD.

## Initial diagnostics to monitor

- group reward variance and mixed-group fraction;
- evidence ready/skipped/error fraction;
- DINO score and crop area fraction from cached `evidence.json` records;
- `actor/seed_opd_teacher_gap_mean`;
- `actor/seed_opd_gate_mean` and gate-active ratio;
- correct/incorrect rollout splits of teacher gap, sampled-token probability
  ratio, and gate mean (diagnostic only; outcome never weights the loss);
- raw and weighted OPD loss;
- `reward/answer_reward_mean`, `reward/format_reward_mean`, and their weighted
  means;
- GRPO loss, policy KL, accuracy, response length, and entropy as a diagnostic
  metric only (its loss coefficient is zero).

## Vision-OPD reward comparison

The sibling Vision-OPD training launcher sets `custom_reward_function.path=null`,
so its released 6K run does not directly enable this project reward. The same verl
fork does contain `verl/utils/reward_score/geo3k.py`, whose `compute_score` uses
the same weighted form `(1 - format_score) * accuracy + format_score * format`.
Our implementation applies that pattern to a generic terminal `FINAL: X` anchor,
returns all components through verl's custom reward interface, and writes them to
the standard rollout JSONL as well as the trainer metrics.

The formal data run is one epoch: 5,928 prompts at a prompt batch of two, or
2,964 updates and 47,424 sampled rollouts. `TOTAL_STEPS=150` remains the
SEED-aligned pilot, while `TOTAL_STEPS=10` is the initial systems and signal
validation. Inspect cached Analyzer outputs and crops for answer leakage and
localization failures before the full run.

## Vision-OPD-6K split and information hygiene

The released Vision-OPD-6K file has 6,241 training records and no official test
split. This project creates a deterministic answer-stratified 95/5 holdout using
seed 42. The test count is rounded upward, yielding 5,928 train and 313 test
records; this also makes the training split exactly divisible by the two-prompt
batch used by verl's `drop_last=True` loader.

The released `images` field is a full image with a red Oracle bounding box, its
question contains a red-box focus hint, and `teacher_images` is an Oracle crop.
Using those as normal inputs would leak localization supervision into the
Student and invalidate the proposed Analyzer/DINO experiment. The conversion
therefore uses only the unboxed `original_images` file and removes the hint. The
released bbox, crop path, and overlay path remain under `extra_info` solely for
offline localization auditing; the online trainer reads only `question` and
`image_path` from that metadata.

## Two-GPU full time sharing

The local Vision-OPD repository already uses verl's hybrid actor/rollout engine
with actor parameter offload, optimizer offload, and reference offload. This
project makes the full release path explicit and adds the settings needed for the
more demanding GRPO + visual-Teacher + reference-KL workload:

| Phase | GPU-resident state | State kept on CPU |
| --- | --- | --- |
| rollout | vLLM weights and KV cache | FSDP actor, optimizer, reference model |
| reference KL scoring | reference FSDP shard | sleeping vLLM, actor/optimizer |
| Student/Teacher scoring and update | actor FSDP shard and optimizer working set | sleeping vLLM, reference model |
| Analyzer and grounding | none | external Analyzer request and GroundingDINO |

`free_cache_engine=true` plus vLLM 0.18 selects sleep level 2, which destroys
the rollout weights and KV cache before training. `enforce_eager=true` disables
CUDA Graph because captured graphs cannot be offloaded at the phase boundary.
`layered_summon=false` is required because that option would force sleep level 1.
The entrypoint checks all of these conditions and aborts instead of silently
running a partial-sharing configuration.

Within the training phase, the actor uses parameter, optimizer, and activation
CPU offload, gradient checkpointing, remove-padding, dynamic batches, fused
Qwen3.5 LM-head computation, PyTorch Fused AdamW, and a per-GPU micro-batch of one. Rollout and
reference log-probability micro-batches are also one. `max_num_seqs=16` avoids
profiling the original 1024-sequence vLLM default for a step containing only
2 prompts x 8 responses. These choices trade PCIe traffic and speed for a lower
and more predictable memory peak.

Fused AdamW is enabled through verl's existing optimizer passthrough rather than
a custom optimizer implementation:

```yaml
actor_rollout_ref:
  actor:
    optim:
      optimizer: AdamW
      optimizer_impl: torch.optim
      override_optimizer_config:
        fused: true
        foreach: false
```

A CUDA smoke test verified two BF16 optimizer steps with the complete state
transition `CUDA -> CPU offload -> CUDA reload`; `step`, `exp_avg`, and
`exp_avg_sq` all moved back successfully before the second fused update.

## Host RAM protection

The current job cgroup already provides the hard boundary requested for this
machine: `/sys/fs/cgroup/memory/memory.limit_in_bytes` is `236223201280`, or
exactly 220 GiB. Ray 2.53 detects the same 220 GiB as node memory. The launcher
adds an earlier Ray OOM-prevention guard so a worker is stopped before the
kernel's cgroup OOM killer selects a process.

The effective Ray threshold is

```text
min(threshold_ceiling, (min(requested_cap, Ray_total) - headroom) / Ray_total)
```

With the defaults on this machine this is `min(0.95, 216 / 220) = 0.95`, so
Ray begins intervention at approximately 209 GiB. The monitor interval is 100
ms, the dashboard is disabled, and the object store is explicitly limited to 8
GiB instead of using Ray's automatic fraction of available memory.

The controls can be adjusted without changing code:

```bash
RAY_NODE_MEMORY_CAP_GIB=220 \
RAY_MEMORY_GUARD_HEADROOM_GIB=4 \
RAY_MEMORY_USAGE_THRESHOLD_CEILING=0.95 \
RAY_OBJECT_STORE_GIB=8 \
RAY_MEMORY_MONITOR_REFRESH_MS=100 \
./scripts/run_visual_seed.sh
```

Ray's threshold is a node-wide, soft OOM-prevention trigger, not an allocation
quota. It counts other processes in the same node/cgroup and may terminate a
Ray worker when crossed. The cgroup is the actual hard 220 GiB boundary. If the
launcher is moved outside this cgroup, use a scheduler/container cgroup limit as
well; the Ray guard by itself cannot make a strict no-overshoot guarantee.
