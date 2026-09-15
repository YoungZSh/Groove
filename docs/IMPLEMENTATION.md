# Implementation decisions

## What comes from each reference

This project uses a sampled-token teacher/student log-probability gap as an
uncentered signed reverse-KL advantage.  It adds that token credit to the GRPO
advantage before one shared PPO/dual-clip objective is evaluated.  The retained
skill is a multimodal Crop/Zoom prefix generated from group-level rollout contrasts.

The project vendors the required Qwen3.5/verl integration under `src/verl` so
the training runtime is part of this repository.  The original full-logit VOPD
KL/JSD path remains available for compatibility, while the `groove` mode
avoids full-vocabulary logits and uses only sampled-token log probabilities.

## Information boundaries

The Analyzer input contains:

- the original image and question;
- reasoning traces programmatically separated into successful and failed lists
  using the semantic evaluator's raw `accuracy` result.

It does not contain rollout IDs, parsed predictions, numeric rewards, or the
ground-truth answer. Training reward shaping is deliberately separate from this
semantic split. For visual-QA runs, an Antidoom-style detector prevents a rollout
from receiving positive reward when one exact contiguous span repeats at least
four times over at least 80 characters. The raw Judge `accuracy` remains unchanged,
and the detected loop suffix is removed before the reasoning is sent to the Analyzer.

Visual-QA training keeps semantic correctness on the original `[0, 1]` scale and
adds a negative-only format term:

```text
accuracy = LLM-Judge semantic correctness in {0, 1}
format_penalty = 0 if one nonempty lowercase <answer>...</answer> pair ends the
                 response (preceding reasoning is allowed), otherwise -1
score = accuracy + 0.2 * format_penalty
```

Thus a strictly formatted correct answer receives `1.0`, while a semantically
correct bare answer receives `0.8`. Analyzer grouping continues to use the raw
`accuracy`, not this shaped `score`.

### Plain reasoning followed by the final answer (next 2B runs)

The response instruction and native chat-template setup live in
`src/groove/response_prompt.py`; `configure_response_format()` installs
`ReasoningAnswerDataset` from `src/groove/reasoning_dataset.py`. Semantic
judging, format shaping, and repetition handling live in
`src/groove/semantic_reward.py`. The Judge environment variables use the
`GROOVE_JUDGE_` prefix, and repetition settings use `GROOVE_REPETITION_`.

The formal launchers are `scripts/run_grpo_2b.sh` and
`scripts/run_grpo_opsd_2b.sh`. Their default prepared-data directories are
`data/vstar_grpo_2200_seed20260904` and
`data/vstar_opsd_2200_seed20260904`. Set `DATA_DIR` to an existing split or
provide a local directory symlink at the default path; each split must contain
`train.parquet` and `validation.parquet`. Existing datasets and archived run
records are preserved, including embedded image paths and source metadata.
The naming change does not alter prompts, rewards, or optimizer settings.

Both formal 2B launchers select `data.response_format=reasoning_answer`. The
Student is instructed to explain its reasoning as ordinary text, then put only
the final answer in one terminal `<answer>...</answer>` pair. No `<think>` or
`<reason>` wrapper is requested, and there is no new reasoning-length reward.
An answer-only completion still has valid answer formatting; reasoning is
requested by the prompt rather than enforced by a separate correctness gate.

`ReasoningAnswerDataset` replaces the system instruction in memory for both
train and validation. Existing parquet files, row order, questions, original
images, reference labels, and Analyzer-only metadata are preserved. Only the
original image and question enter the Student user message. Teacher prompts
continue to derive from the Student prompt and add training-only evidence.

Qwen3.5 uses its native non-thinking mode with `enable_thinking=false`. Before
workers start, the entrypoint reads the model's original chat template unchanged
and shares it between Student and Teacher. The generation prefix includes the
assistant header followed by `<think>\n\n</think>\n\n`, closing the thinking block
before response sampling begins. These prefilled tokens belong to the prompt,
not the sampled response or its token advantages. The base checkpoint files are
not edited, and explicit custom-template overrides are rejected in this mode.
This restores the original template behavior for subsequent runs; the completed
2026-09-06 reasoning experiments used a modified template with the prefill
removed. The plain-reasoning instruction and format reward remain unchanged.

Format validation counts opening and closing answer markers independently,
rejecting nested, duplicated, empty, non-lowercase, or unbalanced tags and text
after the answer. A correct `reasoning + <answer>answer</answer>` receives `1.0`;
a semantically correct malformed completion receives `0.8`. The Judge sees the
extracted answer, and raw `accuracy` and repetition handling are unchanged.

These settings require a fresh experiment name. The historical format-0.2 run
required the whole response to be one answer block and penalized reasoning
outside it; its logged format rate is not comparable to the corrected rule.
The two launchers currently retain different seeds (GRPO: 22, GRPO + OPSD:
20260904), which must be aligned for a controlled comparison.

The separate rule-based `FINAL: X` reward has two independent components:

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
For this rule-based dataset, `accuracy` is still the answer-match bit, but the
Analyzer consumes only the resulting successful/failed membership. It does not
use the weighted terminal `score` as its correctness decision.

A successful rollout makes its answer inferable to the Analyzer; that hindsight
is intentional, while direct answer leakage into Teacher-visible text is not.
The Analyzer's visible instruction is
required to describe an inspection action, not an option or conclusion. Comparative
visual descriptors such as candidate colors or shapes are allowed when they tell
the Teacher what to inspect; explicit answer assertions and option letters are
still rejected. Analyzer messages and GroundingDINO queries use English; the latter
is an explicit tool contract because the deployed DINO checkpoint is English-oriented.
After each visual-tool call, the exact returned crop is sent back to the Analyzer as
an image message so it can verify or revise the query before finalizing the focus
program. Every usable crop receives a candidate ID. After all tool attempts, the
Analyzer selects the best one to three candidates needed for the visual comparison;
there is no IoU deduplication or implicit latest-round preference. Unselected boxes
remain audit-only. Exact grounding phrases stay in the private execution record and
are not Student targets.

### Full-image instance boxes for counting

Counting can opt into `ANALYZER_ENABLE_INSTANCE_BOXES=true`. This exposes the
native `ground_instances(query)` tool in addition to `ground_image` and
`read_text`, with an extra Analyzer instruction to verify all candidate boxes.
It uses the existing DINO detector as a multi-instance baseline. The local
detector returns its full thresholded candidate set; a remote DINO server must
support the `return_all=true` request and return an `instances` array. The
updated `remote_tools/dino_server.py` advertises `ground_instances` in its health
response. An older single-box response is rejected rather than treated as an
exhaustive detection result. The opt-in is disabled by default until the serving
endpoint is updated. Editing the server source locally does not deploy it.

Each instance contains a finite positive-area `bbox` in original-image pixels
and a detector `score`. Geometry is clipped to image bounds, with invalid boxes
failing the result. Candidate order and overlaps are preserved; no implicit NMS,
instance-count gate, or confidence-based advantage weighting is introduced.

One tool result is one candidate image, even when it contains more than three
boxes. The Analyzer sees a preview of the full original image with all candidate
boxes, and selects its candidate ID through the existing one-to-three-image
selection contract. It may combine that view with ordinary detail crops.
`ToolRegion.kind=instance_boxes` and `instances` carry the validated coordinates
to the renderer. The `crops` field retains its historical name in saved evidence;
its selected images now carry `kind=crop` or `kind=instance_boxes`.

The renderer produces a separate native-resolution PNG with thin, unfilled red
rectangles. It does not paint instance IDs, a total count, OCR strings, or answer
text. The original image file is preserved. Teacher input contains the original
image first, then the selected supplementary images in their selected order.
The prompt labels overlays as candidate instance boxes and asks the Teacher to
check misses, duplicate boxes, and false matches against the original image.
The Student still sees only its original image and question. Reward, GRPO,
sampled-token OPSD, reference KL, and optimizer behavior are unchanged.

Complete tool traces are now persisted privately in `TeacherEvidence.tool_trace`
for new ready/error evidence records, including unselected candidates. They are
never inserted into the Teacher prompt or Student input. Historical records
without this field remain readable but do not acquire missing call histories.
The trainer also logs selected instance-box image and box counts as diagnostics;
these are candidate counts, not verified object counts or rewards.

### Independent counting tool with a fixed DINO baseline

`ANALYZER_ENABLE_COUNTING=true` exposes `count_objects(target, region=None)` as
the counting interface. When both counting flags are enabled, the public tool
list exposes `count_objects` instead of the raw `ground_instances` prototype.
The regular `ground_image` and `read_text` tools remain available. Thresholds or
a desired object count are not accepted as tool-call arguments.

The first backend is DINO with a fixed profile in `src/groove/counting.py`:
box threshold 0.35 and text threshold 0.25. Hard NMS is disabled by default:
different occluded objects can have heavily overlapping boxes. The sheep probe
found that IoU 0.5 suppression removed one overlapping candidate and reduced
the ROI result from eight boxes to seven. Preserve such candidates for visual
verification. An explicit NMS profile remains available for offline comparisons;
it does not deduplicate Analyzer candidate images, ordinary crops, or raw
`ground_instances` results.
These values are baseline settings, not a promise of accurate counting on every
image. They must not be fitted to a known answer on an individual example.

For region-restricted questions the Analyzer can supply a rectangle in the
original coordinate system. The tool crops that region before detection, maps
the resulting boxes back into the full-image coordinate system, and uses the
shared unfilled-box renderer. This preserves the global scene in the Teacher's
marked image. Out-of-bounds regions are rejected, not silently clipped: this
lets the Analyzer correct mixed coordinate units rather than count an unintended
partial scene. With a remote detector, only the ROI is sent to the multi-instance
endpoint; temporary ROI files are local and are removed after the request.

The tool returns an estimated count, kept candidate indices, fixed profile,
and original-coordinate boxes for inspection/audit. Only the image and the
answer-neutral focus enter the Teacher prefix. Empty or invalid detections do
not fabricate evidence. The serving DINO endpoint still needs multi-instance
support; the cached local detector can be used for isolated CPU probes without
changing or restarting the deployed services.

## Mask semantics

The following mask semantics apply when the joint objective is enabled with
`OPSD_ENABLED=true`. The default GRPO-only ablation selects vanilla policy loss,
does not construct a self-distillation batch, and therefore has no evidence mask
or Teacher forward pass.

`self_distillation_mask[i] = 1` means that the group containing rollout `i`
has a valid visual Teacher prefix.  Every rollout in that group receives the
same value. It never means that rollout `i` was wrong.  This makes the initial
objective:

```text
all rollout tokens       -> GRPO advantage
tokens in evidence group -> GRPO advantage + lambda * signed OPSD advantage
tokens in fallback group -> GRPO only
```

The default eligibility rule is now simply a successful Analyzer/DINO execution.
Mixed, all-correct, and all-wrong groups can all receive OPD. This keeps auxiliary learning
alive on tied-reward groups where the GRPO advantage is zero. The compatibility
switch `GROOVE_MIXED_GROUPS_ONLY=true` restores the earlier mixed-only probe.

## Reference recipe alignment

The following settings define the reference sampled-token visual-evidence recipe:

| Detail | Reference recipe | This project |
| --- | --- | --- |
| rollout group | 8 | 8 |
| sampled-token credit | positive sigmoid gate | **signed `log p_teacher - log p_student`** |
| OPSD advantage coefficient | 0.01 | 0.01 |
| actor learning rate | 1e-6 | 1e-6 |
| PPO clip | 0.2 | 0.2, explicitly overriding Vision-OPD's user preset |
| PPO epochs | 1 | 1 |
| entropy coefficient | 0.001 in the reference defaults | **0.0 (disabled)** |
| reference KL | low-var KL, coefficient 0.01 | **0.001 (verl default)** |
| GRPO advantage | group standard-deviation normalization | same |
| Analyzer temperature | 0.0 | 0.0 |
| Analyzer completion cap | 2048 for the public visual launcher | 2048 |
| total steps | 150 in the paper | GRPO-only: one 5,928-example epoch (370 updates); 150 is the pilot override |

The default GRPO-only ablation uses prompt batch 16 and PPO mini-batch 128 (one
complete 16x8 rollout batch), micro-batch 1, and parameter/optimizer/reference
offload. The joint launcher keeps its historical prompt batch 32 and PPO
mini-batch 256; the formal supervisor overrides these to 8 and 64. A larger
reference run uses micro-batch 8 and no actor offload on eight A800 80GB GPUs.
The fine-grained VQA data keeps a 4096-token prompt budget rather than a fixed
1024-token budget because its image/question format and multiple visual prefixes
differ.

One algorithmic difference is deliberate. A per-trajectory baseline analyzes each
trajectory separately and creates a text skill. Here the external Analyzer sees
all eight trajectories jointly, performs the requested success/failure contrast,
and produces one shared multimodal Crop/Zoom skill for the group. Teacher and
evidence-advantage targets are detached; the skill construction is group-contrastive.

The regularization choice is also task-specific: the training loss uses a shared
PPO objective over the combined GRPO and signed OPSD advantages, followed by
verl's reference-policy KL. Entropy regularization is disabled, and the KL
coefficient uses verl's `0.001` default instead of the reference `0.01`, because
this is a short single-turn VQA task rather than a long-horizon exploration environment.

## Teacher scoring and refresh behavior

Both scoring passes use the pre-update actor under `no_grad`. Their signed gap
is cached in the combined advantage before any actor optimizer step and stays
fixed across that batch's PPO mini-batches and epochs. The next rollout batch
uses the latest actor for both contexts. There is no separate Teacher optimizer
or EMA update. The external Analyzer and DINO remain fixed in version one.

## Bundled training runtime

The complete runtime needed by the actor, rollout, reference-policy, and Hydra
training paths is vendored under `src/verl`.  The project-specific additions are
therefore ordinary source files rather than a patch applied to a sibling
repository:

- `src/groove/losses.py` contains the shared signed evidence calculation;
- `src/verl/trainer/ppo/core_algos.py` retains a compatibility entrypoint;
- `src/groove/verl_trainer.py` merges evidence credit in its post-advantage hook;
- `src/verl/workers/utils/losses.py` runs the existing PPO and reference KL losses;
- `src/verl/trainer/config/groove.yaml` contains the Hydra preset;
- `src/groove/advantage_metrics.py` reports batch and outcome-group diagnostics.

The existing Conda environment supplies heavyweight runtime dependencies such as
PyTorch, Ray, vLLM, and Transformers; the source tree supplies the matching
Python runtime implementation.

## Initial diagnostics to monitor

- group reward variance and mixed-group fraction;
- evidence ready/skipped/error fraction;
- DINO score and crop area fraction from cached `evidence.json` records;
- signed OPSD teacher-gap mean, spread, and positive/negative token fractions;
- correct/incorrect rollout splits of teacher gap, sampled-token probability
  ratio, and signed advantage mean (diagnostic only; outcome never weights OPSD);
- raw and coefficient-weighted OPSD advantage RMS;
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

The default GRPO-only data run is one epoch: 5,928 prompts at a prompt batch of
16, or 370 complete updates and 47,360 sampled rollouts (`drop_last=True`).
`TOTAL_STEPS=150` remains the reference-aligned pilot, while `TOTAL_STEPS=10` is
the initial systems and signal validation. The GRPO-only path makes no Analyzer
calls; inspect cached Analyzer outputs and crops only for joint OPSD runs.

## Vision-OPD-6K split and information hygiene

The released Vision-OPD-6K file has 6,241 training records and no official test
split. This project creates a deterministic answer-stratified 95/5 holdout using
random state 42. The test count is rounded upward, yielding 5,928 train and 313 test
records; the GRPO-only batch of 16 drops the final eight records because verl's
training loader uses `drop_last=True`.

The released `images` field is a full image with a red Oracle bounding box, its
question contains a red-box focus hint, and `teacher_images` is an Oracle crop.
Using those as normal inputs would leak localization supervision into the
Student and invalidate the proposed Analyzer/DINO experiment. The conversion
therefore uses only the unboxed `original_images` file and removes the hint. The
released bbox, crop path, and overlay path remain under `extra_info` solely for
offline localization auditing; the online trainer reads only `question` and
`image_path` from that metadata.

## Two-GPU full time sharing

The bundled verl runtime provides the hybrid actor/rollout engine with actor
parameter offload, optimizer offload, and reference offload. This project makes
the full release path explicit and adds the settings needed for the more
demanding GRPO + visual-Teacher + reference-KL workload:

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
./scripts/run_groove.sh
```

Ray's threshold is a node-wide, soft OOM-prevention trigger, not an allocation
quota. It counts other processes in the same node/cgroup and may terminate a
Ray worker when crossed. The cgroup is the actual hard 220 GiB boundary. If the
launcher is moved outside this cgroup, use a scheduler/container cgroup limit as
well; the Ray guard by itself cannot make a strict no-overshoot guarantee.
