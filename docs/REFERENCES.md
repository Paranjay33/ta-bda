# References

## Data and models
- Patwa, P., Aguilar, G., Kar, S., Pandey, S., PYKL, S., Gambäck, B., Chakraborty, T., Solorio, T.,
  & Das, A. (2020). *SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets.*
  Proceedings of the 14th Workshop on Semantic Evaluation (SemEval-2020).
- Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.
  Checkpoint: `FacebookAI/xlm-roberta-base` (Hugging Face Hub).
- Nayak, R., & Joshi, R. (2022). *L3Cube-HingCorpus and HingBERT: A Corpus and BERT Models for
  Code-Mixed Hindi-English Language.* Checkpoint: `l3cube-pune/hing-roberta` (Hugging Face Hub).
- Singh, P., & Lefever, E. (2020). *Sentiment Analysis for Hinglish Code-mixed Tweets by means of
  Cross-lingual Word Embeddings.* (cited in the midterm literature review)

## Methods
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural
  Networks.* ICML — temperature scaling, ECE.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines…* — margin calibration
  (evaluated and rejected for E2, see `docs/EXPERIMENTS.md`).
- Porter, M. F. (1980). *An algorithm for suffix stripping.* — the stemmer applied upstream to the data.
- McNemar, Q. (1947); Efron, B., & Tibshirani, R. (1993) — paired significance test and bootstrap CI.
- Dettmers, T., et al. (2022). *8-bit Optimizers via Block-wise Quantization.* ICLR — 8-bit AdamW.

## Software
- Apache Spark MLlib 3.5.1 — https://spark.apache.org/docs/3.5.1/ml-guide.html
- Hugging Face Transformers 4.46 — https://huggingface.co/docs/transformers
- PyTorch 2.5.1, scikit-learn 1.9, NLTK 3.10 (PorterStemmer, NLTKWordTokenizer, English stop-words),
  bitsandbytes 0.50, FastAPI, SQLAlchemy 2, Alembic, Playwright.

Bibliographic details were written from the project's midterm references and general knowledge;
verify page numbers/venues against the live sources before citing in the final report.
