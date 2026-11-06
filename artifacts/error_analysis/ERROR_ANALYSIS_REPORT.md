# Error analysis (locked test split, machine-generated)

Categories are **heuristic hints** (regex/lexicon), not manual annotations, and do not claim to explain why a model erred. Text is the preprocessed (lowercased, stemmed, stopword-filtered) form, so English negators such as `not`/`no` and the contrast word `but` were removed upstream -- the negation/contrast hints therefore only fire on Hindi markers (`nahi`, `mat`, `lekin`, `par`, ...).

| Exp | Error rate | negation (n / err) | contrast (n / err) | emoji (n / err) | code_mixing (n / err) | short_text (n / err) |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 50.2% | 387 / 56.1% | 62 / 54.8% | 279 / 53.0% | 736 / 50.8% | 33 / 48.5% |
| E2 | 45.0% | 387 / 48.1% | 62 / 41.9% | 279 / 47.7% | 736 / 45.7% | 33 / 42.4% |
| E3 | 35.3% | 387 / 40.6% | 62 / 35.5% | 279 / 34.1% | 736 / 38.7% | 33 / 27.3% |
| E4 | 32.2% | 387 / 35.7% | 62 / 29.0% | 279 / 32.6% | 736 / 35.1% | 33 / 27.3% |

## Most frequent confusions

- **E1**: positive->neutral (176), neutral->negative (147), neutral->positive (144)
- **E2**: positive->neutral (151), neutral->positive (141), neutral->negative (136)
- **E3**: neutral->negative (139), positive->neutral (112), negative->neutral (106)
- **E4**: neutral->negative (125), neutral->positive (116), negative->neutral (94)
