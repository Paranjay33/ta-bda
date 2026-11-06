# Legacy midterm artifacts (preserved, not used)

Moved here unchanged on 2026-11-06 so the new locked evaluation cannot overwrite them.

- `e3_train_manifest.json` — midterm E3 run. Its "validation" metrics (acc 0.7402, Macro-F1 0.7410)
  were measured on the supplied validation file, 1,446 of whose 3,000 rows are copies of training rows.
- `e3_locked_test_metrics.json`, `e3_test_predictions.csv`, `e3_confusion_matrix.png` — computed on
  a 2-row `test.txt`; meaningless as test results.
- The midterm E3 weights remain in `artifacts/models/e3/{best,checkpoint-3644,checkpoint-4555}`
  (untouched; not registered for serving).
- No E4 midterm artifact ever existed in this workspace.
