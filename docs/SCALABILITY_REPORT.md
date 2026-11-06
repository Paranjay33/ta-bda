# BDA scalability report

**Measured** on 2026-11-07 with `bash scripts/run_scalability_benchmark.sh`. Raw data:
`artifacts/scalability/spark_scalability_benchmark.csv`,
`artifacts/scalability/python_local_baseline_scalability.json`, log `artifacts/logs/scalability.log`.

## Setup

- Workload: the E1 pipeline (RegexTokenizer → CountVectorizer(20k) → IDF → multinomial LR) fitted on
  the processed training split, then applied to every row (`fit` + `transform().count()`).
- Scales 1× / 5× / 10× are made by **replicating** the 11,654 training rows. Replicated rows are a
  synthetic workload for runtime measurement only — not new observations; no accuracy is reported.
- Spark 3.5.1, **local mode** `local[*]` on one laptop (16 logical cores, Ryzen 7 7435HS, 15.8 GB RAM,
  JVM heap 3 GB, Java 17). **No cluster was used.**
- One untimed warm-up run first (JIT/codegen: the warm-up took 17.9 s vs ~7.4 s for timed 1× runs),
  then 2 timed repetitions per scale; replicated data persisted MEMORY_AND_DISK.
- Reference: scikit-learn `TfidfVectorizer(token_pattern=\S+, 20k)` + `LogisticRegression`, single
  process, same rows/scales, same warm-up/repeat policy.
- The machine was otherwise idle apart from the user's desktop applications.

## Results (median of 2 repetitions)

| Scale | Rows | Spark runtime (s) | Spark rows/s | Partitions | Spark stages / tasks (per run) | scikit-learn runtime (s) |
|---:|---:|---:|---:|---:|---|---:|
| 1× | 11,654 | 7.45 (7.67, 7.23) | 1,565 | 1 | 116 / 116 | 1.13 |
| 5× | 58,270 | 7.54 (8.47, 6.61) | 7,725 | 5 | 117 / 581 · 96 / 476 | 3.99 |
| 10× | 116,540 | 24.67 (24.22, 25.13) | 4,724 | 10 | 209 / 1,265 · 225 / 1,361 | 6.63 |

Stage/task counts come from a `SparkListener` registered during each timed run.

## Interpretation

- **1× → 5×:** runtime stays flat (~7.5 s) while rows grow 5×. The replicated DataFrame has 5
  partitions, so tasks run in parallel on the local cores; at 1× the job is dominated by fixed
  per-stage overhead (~116 stages for ~100 LR iterations).
- **5× → 10×:** runtime triples and the stage count roughly doubles (≈ 210–225). With a 3 GB heap the
  10× persisted data and per-iteration aggregations no longer stay comfortably in memory (spill to
  disk is likely; GC was not instrumented). This is a memory limit of the single-machine setup, not a
  property of a cluster.
- **Spark vs scikit-learn:** on a single machine with ≤ 117k short texts, scikit-learn is 2.5–6.6×
  faster. The data fits in RAM, so Spark's job scheduling and serialization costs dominate. Spark's
  value here is the programming model (the same pipeline would run unchanged on a cluster via
  `SPARK_MASTER`) and horizontal scaling beyond one machine's memory, which **was not measured**.

## Limitations

Two repetitions only; laptop timings vary (5× runs differ by 1.9 s). No cluster run, no Spark UI
screenshots, no GC/spill metrics. A first attempt on 2026-11-07 failed at 10× with a JVM native
out-of-memory error while the system was memory-starved (a game plus GPU training running); it also
exposed a benchmark bug (the cached base data was evicted after the 1× run), which was fixed before
these measurements.
