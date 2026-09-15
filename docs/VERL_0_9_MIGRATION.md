# VERL 0.9 migration

## Baseline

- Previous vendored version: `0.7.0.dev`
- New stable baseline: VERL `v0.9.0`
- Upstream commit: `483b8a009ba3a97563edee3a19887e4862b8094a`
- The downloaded upstream checkout is kept locally under
  `TMP/upstream/verl-v0.9.0` and excluded from Git.

The complete upstream `verl/` package was used as the new `src/verl/` base.
Project behavior is layered on top instead of retaining stale upstream files.

## Project adaptations reapplied on top of 0.9

1. `groove.verl_entrypoint` uses the v0.9 synchronous TaskRunner building
   blocks and instantiates `GrooveRayPPOTrainer` directly.
2. The launcher uses v0.9's prompt-level `ppo_mini_batch_size` semantics. A
   Batch-16, `n=8` step sets the value to 16; VERL expands it to 128
   completions internally.
3. Semantic judging uses the v0.9 reward-loop interface. A single asynchronous
   reward actor accepts concurrent trajectory calls and dispatches blocking
   HTTP requests through its executor, preserving rollout/judge overlap without
   spawning dozens of Ray actors.
4. `RLHFDataset` retains the project `image_max_pixels` compatibility layer so
   Qwen smart-resize limits are attached without mutating source rows.
5. GRPO group statistics retain the float64 uniform-group fix, preventing
   identical non-binary rewards such as `0.1` from producing numerical credit.
6. The legacy actor-local OPSD implementation was migrated to a trainer-level
   post-advantage hook. When GROOVE is enabled, privileged visual prompts are
   built once per UID, scored by the current actor before its update, converted
   to signed sampled-token advantages, and added to GRPO advantages. When it is
   disabled, the hook returns before all evidence and Teacher construction.
7. The Qwen3.5 non-thinking constraint remains explicit in the data and launch
   configuration. The experiment launcher rejects attempts to override it with
   `enable_thinking=true`.

## Qwen3.5 packed-sequence correctness

VERL v0.9 contains the upstream packed-sequence fix from PR #6660. It forwards
`cu_seqlens` and `seq_idx` through Qwen3.5 Gated DeltaNet and causal-convolution
layers so independent samples do not share recurrent state. The 2B launcher can
therefore use `MODEL_USE_REMOVE_PADDING=true` again.

## Validation

- Project unit tests: 44 passed.
- Three-step end-to-end probe: the archived log under `outputs/logs/` matching
  `*verl090-rmpad-3step-probe-v3.log`.
- Rollout-vs-actor mean probability difference by step:
  `0.00701`, `0.00665`, `0.00640`.
- Pearson correlation by step:
  `0.99931`, `0.99940`, `0.99940`.
- Step wall time after warmup: 67 seconds and 50 seconds.
- Actor update time after warmup: about 6 seconds.
- Response truncation: 0% for all three steps with a 1024-token limit.

The formal training job was not restarted during migration.

## Remaining architectural note

GROOVE currently selects `trainer.use_v1=false` because its privileged
current-policy Teacher needs the post-advantage extension point in the
synchronous trainer. The code is from the v0.9 stable package, but this specific
trainer path is marked legacy upstream. Moving the hook to VERL's TransferQueue
v1 trainer is a separate refactor and is not required for the current GRPO run.
