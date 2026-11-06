# Dataset card — SentiMix Hinglish (as supplied to this project)

| Field | Value |
|---|---|
| Source | SemEval-2020 Task 9: Sentiment Analysis of Code-Mixed Tweets, Hinglish track (Patwa et al., 2020) |
| Obtained as | `FinalTrainingOnly.tsv`, `ValidationOnly.tsv`, `FinalTest.tsv`, supplied by the project team (exact download origin not recorded in the workspace) |
| Language | Hindi-English code-mixed, Roman script (Hinglish), some English-only and emoji |
| Unit | one tweet (sentence-level label) |
| Labels | negative / neutral / positive (raw values 0 / 1 / 2) |
| Labelled rows supplied | 17,594 (14,594 + 3,000); 3,000 unlabelled official-test rows |
| Used after cleaning | 14,568 unique texts → train 11,654 · validation 1,457 · test 1,457 |
| Class balance (used) | negative 29.3% · neutral 37.5% · positive 33.2% |
| Text length | median 16 tokens, 95th percentile 23, max 109 (supplied training file) |
| Integrity | raw and processed SHA-256 in `data/processed/split_manifest.json` |

## Preprocessing applied before this project (not reversible)

Lowercasing, NLTK English stop-word removal (incl. `not`, `no`, `but`), Porter stemming of every
token, a trailing ` nan` on ~44% of rows, and `@ user` mentions with a space. These were
detected statistically, not documented by the supplier.

## Preprocessing applied by this project

Strip the ` nan` artefact → shared normalization (HTML unescape, URL/mention masking, whitespace
collapse) → drop empty rows (26) → drop texts with conflicting labels (114 texts / 284 rows) →
collapse exact duplicates (2,716 rows) → stratified 80/10/10, seed 42.

## Known issues

- The supplied validation file overlapped the training file (1,446 identical rows); hence the
  re-split. Any result computed on the supplied split (including the midterm figures) is inflated.
- Label noise: SentiMix neutral is known to be noisy; 114 texts appeared with conflicting labels.
- Topic skew: much of the data concerns Indian politics, religion, cricket and film; it contains
  abusive language and slurs. Models can learn topic or identity-term correlations.
- No user identifiers: near-duplicate tweets by the same user may sit in different splits.
- Official test gold labels are not present, so SemEval-leaderboard comparability is not possible.

## Licence and redistribution

Use is governed by the SemEval-2020 Task 9 data terms (not re-verified in this project). The data
is not committed to version control, not baked into Docker images and not exposed by the API
(the API never returns training rows).
