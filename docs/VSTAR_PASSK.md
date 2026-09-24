# V*Bench sampled pass@8 comparison

`scripts/evaluate_vstar_passk.py` evaluates the base Student and the DAPO
step-130 best snapshot with the current training-time semantic protocol.
It does not change `scripts/evaluate_vstar.py`, which retains the historical
greedy/rule-only protocol.

The sampling defaults are temperature 1.0, top-p 1.0, top-k -1, repetition
penalty 1.0, eight responses per question and 1024 response tokens. All 191
benchmark questions and their original image bytes are retained. Both models
use the same `ReasoningAnswerDataset` adapter, native non-thinking chat
template, empty think prefill and image processing. Student inputs contain
only the image, question and options; the reference answer is sent only to
the semantic Judge. CUDA Graph is disabled (`enforce_eager=true`).

For the four-GPU comparison, GPUs 0/1 hold two TP=1 base replicas and GPUs 2/3
hold two TP=1 best-model replicas. Each pair receives the same two deterministic
question shards. Each question uses `seed + original_row_index * n`; vLLM
generates n sampled completions. Sampling is not the earlier temperature-0
validation protocol, so sampled pass@1 is reported separately from greedy
accuracy. Each model produces 191 × 8 = 1528 responses.

`generate` writes question-level prompt audits, raw responses, token IDs,
finish reasons and a manifest. `judge` interleaves the models' requests and
scores every answer through `groove.semantic_reward.compute_score` with
`data_source=vstar_bench`. Accuracy stays binary and unshaped; format,
repetition and historical rule matching are diagnostics. Judge failure does
not silently count as an incorrect answer or reduce the denominator.

The current dataset adapter permits either an option letter or answer text in
the final answer tags. Before grading, the semantic adapter removes Student
format instructions from the question and supplies the reference as both letter
and text (for example `(C) purple`). Existing parquet files and evaluation logs
are preserved; comparisons across this correction require a common scoring protocol.

`src/groove/passk.py` requires exactly n distinct sample indices for each
question, full question coverage and matching model inputs. With n=8,
pass@8 is the fraction of questions with at least one correct answer. For
k=1,2,4 it uses `1 - C(n-c,k)/C(n,k)`, where c is the number of correct
responses. Sampled pass@1 therefore equals the mean accuracy across all eight
responses per question. Category summaries, the 0–8 correct-response histogram,
and paired question gains/losses are also retained.

The experiment-specific supervisor under `outputs/diagnostics/` records the
existing service process identities, pauses only the four authorized local
services, runs generation and scoring, and restores the original service
configuration in a finally block. Checkpoints and historical evaluations are
preserved; model merging writes to a new directory. The manifest records data,
model and evaluator hashes. No W&B history is rewritten.
