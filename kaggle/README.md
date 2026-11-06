# Training v3 on Kaggle (E3 + E4)

Why: Kaggle's 16 GB GPU allows the normal fp32 AdamW optimizer, up to 30 epochs and a small
learning-rate search — the 4 GB laptop could not. E3 and E4 get **exactly the same procedure**,
so the E3-vs-E4 research comparison stays fair. The test split is never touched on Kaggle.

## 1. On your laptop — build the upload bundle
```powershell
cd "D:\SEMESTER V\TA-BDA\hinglish-sentiment\hinglish-sentiment"
.\.venv\Scripts\python.exe scripts\make_kaggle_bundle.py
```
→ `dist\hinglishsense_kaggle_bundle.zip` (code + configs + frozen split, a few MB).

## 2. On kaggle.com — upload it as a PRIVATE dataset
Datasets → **New Dataset** → drag the zip → title `hinglishsense-kaggle-bundle` →
visibility **Private** (it contains SentiMix tweets) → Create.

## 3. Create the notebook
Code → **New Notebook** → File → **Import Notebook** → choose `kaggle\hinglishsense_v3_train.ipynb`.
In the right panel:
- **Add Input** → Your Datasets → `hinglishsense-kaggle-bundle`
- Settings → Accelerator → **GPU P100** (or GPU T4 x2 — the notebook uses one GPU either way)
- Settings → **Internet → On** (downloads the two base checkpoints from Hugging Face)

## 4. Run it in the background
**Save Version → Save & Run All (Commit) → Save.** You can close the browser.
Expected time ≈ 2–3 hours (well inside the 12 h limit). Progress: open the version → Logs.

## 5. Download the result
Open the finished version → **Output** → download `hinglishsense_v3_models.zip` (~2.2 GB).
`v3_search_summary.json` shows every run's validation score.

## 6. Back on your laptop — import, evaluate once, serve
```powershell
cd "D:\SEMESTER V\TA-BDA\hinglish-sentiment\hinglish-sentiment"
$env:PYTHONPATH="python\src"
.\.venv\Scripts\python.exe -m hinglish.import_kaggle_v3 "$HOME\Downloads\hinglishsense_v3_models.zip"
```
Then in Git Bash:
```bash
bash scripts/evaluate_v3_locked.sh      # one locked test evaluation + reports + model cards
bash scripts/start_local.sh             # the app now serves the v3 E3/E4 models
```
The import refuses anything trained on a different split, corrupted downloads and wrong label
orders, and archives the v2 results first (`artifacts/metrics/runs/`), so both runs stay visible.

## If something goes wrong
| Problem | Fix |
|---|---|
| "No GPU visible" | Settings → Accelerator → GPU, then re-run |
| Hugging Face download error | Settings → Internet must be **On** (phone-verified account) |
| "Frozen split verification failed" | re-upload the bundle zip unchanged |
| Session stopped half-way | re-run: finished runs are skipped automatically (interactive session only) |
| Out of GPU quota | Kaggle gives ~30 GPU h/week; wait for the weekly reset |
