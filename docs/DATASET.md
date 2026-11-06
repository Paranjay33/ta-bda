# Dataset: SemEval-2020 Task 9 SentiMix Hinglish (supplied release)

## What is in the workspace

The real benchmark data was supplied by the project team as three TSV files, now copied
(not moved — the originals stay at the repo root) to `data/raw/sentimix_supplied/`:

| File | Rows | Schema | Used for |
|---|---:|---|---|
| `FinalTrainingOnly.tsv` | 14,594 | `uid \t text \t label` (no header) | pooled labelled data |
| `ValidationOnly.tsv` | 3,000 | same | pooled labelled data |
| `FinalTest.tsv` | 3,000 | `uid \t text` (**no labels**) | unlabelled batch demos only |

`hinglish.dataset.detect_schema` recognises this shape explicitly (`tsv_uid_text_label` /
`tsv_uid_text`); a file matching no known schema raises instead of being coerced. The original
CoNLL (`meta \t uid \t label` + token lines) and CSV-with-header parsers are kept.

Raw label values `0/1/2` map to canonical names through `dataset.raw_label_aliases` in
`configs/config.yaml` (`0→negative, 1→neutral, 2→positive`); `configs/label_mapping.json`
remains the only name→id mapping (`negative=0, neutral=1, positive=2`). The alias semantics
were checked by reading random samples of each class.

## This is a preprocessed release

The text is not raw tweet text. It was lowercased, filtered with the NLTK English stop-word
list and Porter-stemmed before it reached this project (evidence in `docs/INITIAL_AUDIT.md`).
English negators and contrast words (`not`, `no`, `nor`, `but`, `don't`) were removed upstream;
Hindi ones (`nahi`, `mat`, `lekin`, `par`) survive because they are not English stop-words.
About 44% of rows end in a literal ` nan` (a NaN concatenated upstream). Original spacing,
casing, URLs and mentions cannot be recovered.

## Why the supplied split was not used

The project policy is "use the official split if it is supplied **and verified**". It failed
verification: 1,446 of the 3,000 validation rows are exact copies of training rows (same uid,
text and label), 1,458 share their text with a training row, and uids collide across files
with different texts. Validation scores on that split are inflated, so the documented
fallback was applied.

## The frozen split (`python -m hinglish.prepare_data`)

1. Pool both labelled files (17,594 rows).
2. Strip the trailing ` nan` artefact, then apply the shared normalization contract
   (`docs/PREPROCESSING.md`) → `normalized_text`.
3. Drop 26 rows that are empty after cleaning.
4. Group by exact `normalized_text`: drop the 114 texts (284 rows) whose copies carry
   conflicting labels; collapse the remaining 2,716 duplicate rows to one representative each,
   so no text can appear in two splits.
5. One stratified 80/10/10 split with seed 42.

| Split | Rows | negative | neutral | positive |
|---|---:|---:|---:|---:|
| train | 11,654 | 3,410 | 4,375 | 3,869 |
| validation | 1,457 | 426 | 547 | 484 |
| test (locked) | 1,457 | 426 | 547 | 484 |

Exact-text overlap between every pair of splits is 0 (asserted in code and in tests).

Outputs (`data/processed/`): `train.tsv`, `validation.tsv`, `test.tsv` with columns
`sample_id, source_file, source_row_id, raw_text, normalized_text, label, label_id, split`, and
`split_manifest.json` (raw-file SHA-256, processed-file SHA-256, per-split sample-id hashes,
label distributions, drop counts, preprocessing fingerprint, label-map hash).

`sample_id` is a **generated** identifier (`hs-` + first 16 hex chars of SHA-256 of the
normalized text). The supplied uids are kept as `source_row_id` but are not unique across files.

### Freezing

Re-running `prepare_data` rebuilds the split and compares sample-id hashes with the frozen
manifest: identical → it only verifies file hashes; different → it **refuses** to overwrite
unless `--force-new-split` is passed. `--verify` never writes. Every training and evaluation
entry point verifies the split before running.

## Limitations

- Only one labelled source exists; the test split is drawn from the same distribution as the
  training data (tweets from the SentiMix collection). Official-test comparability with the
  SemEval leaderboard is not possible without the official test gold labels.
- Exact-duplicate removal does not catch near-duplicates (e.g. retweets differing by one token).
- No user/thread metadata exists, so group-aware splitting beyond exact text is impossible.

## Redistribution

SentiMix data must not be committed or published. `data/raw/**` and `data/processed/*.tsv` are
git-ignored and excluded from Docker images (mounted at runtime instead).
