# Limitations

No metric, prediction, timing or deployment in this repository is fabricated. Where something
was not run, it says so.

## Data and evaluation
- **Preprocessed source text.** The supplied SentiMix release was lowercased, stop-word-filtered
  (removing English `not`, `no`, `but`, …) and Porter-stemmed upstream. Every model therefore
  learned from text without English negators/contrast words; live input is mapped into the same
  space (`docs/PREPROCESSING.md`), which means sentences whose meaning hinges on English `not`/`but`
  lose that signal. Hindi negators (`nahi`, `mat`) and contrast words (`lekin`, `par`) survive.
- **Not comparable with the midterm.** The midterm's E3 74.10% was a validation score on a split
  where 48% of validation rows were copies of training rows; the E4 85.16% cannot be traced to
  any artifact. The re-run uses a verified split and slightly different (documented) training
  settings, so the new numbers replace — and should not be compared with — the midterm table.
- **E3 may be under-trained at the shared 5-epoch cap.** Its validation Macro-F1 was still rising
  (0.631 → 0.634 at epochs 4 → 5) while E4 peaked at epoch 1. The cap is part of the matched
  protocol, but a longer budget for both arms could narrow (or widen) the +2.98 pp gap.
- **The midterm showcase sentence** (`movie toh acchi thi but ending bakwas thi`) is predicted
  *neutral* by both E3 and E4: after the upstream preprocessing the models see
  `movi toh acchi thi end bakwa thi`, with the contrast word removed.
- **One split, one seed.** No seed variance or cross-validation; the E3/E4 bootstrap CI covers
  test-sample uncertainty only.
- **In-distribution test.** The test split comes from the same SentiMix collection. Official
  SemEval test labels were not available, so leaderboard comparison is impossible.
- **Exact-duplicate dedup only.** Near-duplicates (retweets with small edits) may cross splits.
- **Error analysis uses heuristic hints**, not manual annotation, and does not explain causes.

## Models
- E1 is unregularised (midterm definition kept); its probabilities are extremely over-confident
  and are labelled uncalibrated in the UI and API.
- E2 scores are temperature-scaled SVM margins (argmax-preserving), not native probabilities.
- E3/E4 were trained with 8-bit AdamW (memory constraint) and `.bin` checkpoints; both arms
  identically.
- Calibration (temperature) is fitted on 1,457 validation tweets; ECE is reported on test.

## Engineering / environment
- **Docker not available** on the build machine: `compose.yaml` and the Dockerfiles are
  prepared but were not built or run. No cloud deployment was performed.
- Spark ran in **local mode** on one laptop; nothing is a multi-node cluster measurement.
- The Spark serving path adds ~0.3–1 s per request (DataFrame creation + job scheduling on
  `local[2]`); acceptable for a demo, not for high-throughput serving.
- Transformer training shared a 4 GB GPU with another application during the session, which
  inflated `training_runtime_seconds`; treat runtimes as environment-specific.
- Background batch jobs run in API threads (no external queue); a restart during a large job
  leaves it `running` — re-submit it.
- Rate limiting is in-memory per process; use a proxy/gateway limiter for multi-replica setups.
- Drift monitoring is descriptive only (class mix, usage, latency, input length over time).

## Scope
HinglishSense is a research/teaching prototype. It does not understand people, sarcasm or
context reliably and must not be used for decisions about individuals.
