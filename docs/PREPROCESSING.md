# Preprocessing contract

There is exactly one implementation of each step, shared by every branch and by serving.

```
TRAINING DATA (already lowercased/stop-word-filtered/stemmed upstream)
  raw_text -> strip trailing " nan" artefact -> shared normalization -> normalized_text
           (hinglish.prepare_data)              (hinglish.normalization)
           -> data/processed/*.tsv  -> consumed AS-IS by E1/E2 (Scala) and E3/E4 (Python)

LIVE / BATCH INPUT (raw user text)
  text -> shared normalization -> dataset-style transform -> model input (same for E1-E4)
          (hinglish.normalization)  (hinglish.dataset_style)
```

## 1. Shared normalization (`python/src/hinglish/normalization.py`, version `shared-norm-v1`)

Unchanged from the midterm design: HTML-entity unescape; URLs → `<URL>`; `@mentions` → `<USER>`;
whitespace collapsed; lowercasing off by default; emoji, repeated characters, punctuation,
negation, slang and code-switching are never removed or translated.

## 2. Cross-branch parity

The midterm described one shared pipeline, but the Scala branch had no normalization at all.
Rather than re-implementing the rules in Scala (two implementations can drift), the Spark
branch now reads the processed TSV and uses its `normalized_text` column verbatim
(`DataIngestion.loadProcessedSplit`). At serving time the Python API sends the already-
transformed text to the Spark service. Parity is therefore by construction, and is tested:

- `python/tests/test_data_contract.py::test_frozen_split_files_match_manifest_and_contract`
  recomputes `normalize(strip_nan(raw_text))` for sampled rows of every split;
- `spark/src/test/.../ServingAndContractSpec` checks the Scala reader keeps `normalized_text`
  byte-for-byte (including quotes) and rejects label/label_id disagreement.

## 3. Dataset-style transform (`python/src/hinglish/dataset_style.py`, `dataset-style-v1`)

Because the supplied text was preprocessed upstream, live text must be mapped into the same
space, or every model sees out-of-distribution tokens (`movie` was never seen — only `movi`).

Steps: lowercase → NLTK `NLTKWordTokenizer` (the tokenizer behind `word_tokenize`, which
reproduces the stand-alone `n't`/`'s` tokens seen in the data) → drop tokens in the NLTK English
stop-word list (`configs/dataset_compat/nltk_english_stopwords.txt`, 198 words, vendored so no
corpus download is needed) → Porter-stem every remaining token. `<URL>`/`<USER>` placeholders are
protected.

Example: `movie toh acchi thi but ending bakwas thi` → `movi toh acchi thi end bakwa thi`.

This is a **reconstruction** of an undocumented upstream step, not a choice of this project.
Its consequence — that English negation/contrast words are invisible to every model — is a
property of the data and is stated in the UI ("What the model actually read"), the model cards
and `docs/LIMITATIONS.md`. Applying the transform to already-processed validation text leaves
>85% of tokens unchanged (test `test_dataset_style_is_near_idempotent_on_training_text`).

It is applied only to raw input (API, batch); it is never re-applied to the processed splits.

## 4. Model-specific tokenization

E3/E4: the checkpoint's own tokenizer, `max_length=128`, truncation, dynamic padding
(`hinglish.tokenization`). E1/E2: Spark `RegexTokenizer(\s+)` (+ `CharacterSplitter`→`NGram(3)`
for E2), fitted as part of the saved `PipelineModel`.

## 5. Versioning

`preprocessing.version` and the dataset-style version are recorded in `split_manifest.json`
(as a SHA-256 fingerprint of the whole preprocessing config), in every model manifest, and in
every API response (`preprocessing_version`).
