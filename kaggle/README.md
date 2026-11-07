# Training E3 + E4 on Kaggle (v3 / v3b)

Kaggle's 16 GB GPU allows full fp32 AdamW and a learning-rate search. E3 and E4 always get
**exactly the same procedure**, so the E3-vs-E4 comparison stays fair. The test split is never
touched on Kaggle.

| Notebook | Overlay | Max epochs | Notes |
|---|---|---|---|
| `hinglishsense_v3_train.ipynb` | `configs/kaggle_v3.yaml` | 30 | first run; LR schedule spread over 30 epochs, so early-stopped runs never reach the decay phase |
| `hinglishsense_v3b_train.ipynb` | `configs/kaggle_v3b.yaml` | 8 | **recommended**: warmup + decay fit the epochs actually used |

Both: LR search {1e-5, 2e-5, 3e-5} on validation (seed 42) → best LR re-run with seeds 42/43/44 →
served model = best validation seed. Early stopping patience 3 on validation Macro-F1.

## 1. Laptop — build the upload bundle
```powershell
cd "D:\SEMESTER V\TA-BDA\hinglish-sentiment\hinglish-sentiment"
.\.venv\Scripts\python.exe scripts\make_kaggle_bundle.py
```
→ `dist\hinglishsense_kaggle_bundle.zip`

## 2. Kaggle — update the dataset
Datasets → your `hinglishsense-kaggle-bundle` → **New Version** → upload the new zip → Create.
(First time: New Dataset, **Private** — it contains SentiMix tweets.)

## 3. Kaggle — notebook
Code → New Notebook → File → **Import Notebook** → `kaggle\hinglishsense_v3b_train.ipynb`.
Right panel: **Add Input** → `hinglishsense-kaggle-bundle` (latest version) · Accelerator **GPU P100**
(or T4 x2) · **Internet On**. Then **Save Version → Save & Run All (Commit)**. ≈1.5–2.5 h.

## 4. Download
Finished version → **Output** → `hinglishsense_v3b-kaggle_models.zip` (≈2.2 GB).
(The earlier 30-epoch run gives `hinglishsense_v3_models.zip` — import it too if you have it.)

## 5. Laptop — install, select (validation only), evaluate once, serve
```powershell
cd "D:\SEMESTER V\TA-BDA\hinglish-sentiment\hinglish-sentiment"
$env:PYTHONPATH="python\src"
.\.venv\Scripts\python.exe -m hinglish.import_kaggle_v3 "$HOME\Downloads\hinglishsense_v3b-kaggle_models.zip"
.\.venv\Scripts\python.exe -m hinglish.import_kaggle_v3 "$HOME\Downloads\hinglishsense_v3_models.zip"   # optional
.\.venv\Scripts\python.exe -m hinglish.select_transformer_run            # shows v2 / v3 / v3b on VALIDATION
.\.venv\Scripts\python.exe -m hinglish.select_transformer_run --apply    # switches only if a Kaggle run is better
```
If it switched, in **Git Bash**:
```bash
bash scripts/evaluate_v3_locked.sh   # ONE locked test evaluation of the selected run + reports
bash scripts/start_local.sh          # serve it
```
If v2 stays best, nothing is switched; the Kaggle runs remain installed and their validation
results (stability across learning rates and seeds) go into the report as a robustness study.

## Safety built in
- Import refuses: a different split, corrupt downloads (SHA-256), wrong label order, overwriting a run.
- Select uses **validation only** (mean of E3 and E4 → both from the same protocol); archives v2 results first.
- `evaluate_v3_locked.sh` refuses to evaluate the same model version twice.

## Troubleshooting
| Problem | Fix |
|---|---|
| "No GPU visible" | Accelerator → GPU, re-run |
| Hugging Face download error | Internet must be **On** |
| "kaggle_v3b.yaml missing" | attach the **latest** dataset version (step 2) |
| "Frozen split verification failed" | re-upload the bundle zip unchanged |
| Out of GPU quota | ~30 GPU h/week; wait for the weekly reset |
