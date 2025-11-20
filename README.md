# Regression_Medical

This repository collects notes for reproducing fairness-aware risk-adjustment analyses using the CMS Synthetic Data set and related JAMA Health Forum study materials.

See [`docs/reproduction_plan.md`](docs/reproduction_plan.md) for detailed steps and slide-ready explanations.

## Quickstart: run the constrained regression reproduction
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure your Kaggle API credentials are configured (typically by placing `kaggle.json` under `~/.kaggle/` or exporting `KAGGLE_USERNAME`/`KAGGLE_KEY`) and that you have accepted the [dataset terms](https://www.kaggle.com/datasets/anikannal/cms-synthetic-data).

3. Run the reproduction script against a CSV from the Kaggle CMS Synthetic Data set (e.g., `Beneficiary_Summary_File.csv` inside the dataset zip):
   ```bash
   python analysis/reproduce.py Beneficiary_Summary_File.csv \
     --outcome TOTAL_SPENDING \
     --features AGE SEX RACE HCC_SCORE DUAL_STATUS DISABILITY_STATUS \
     --group-columns SEX RACE
   ```
   The script downloads the file via `kagglehub`, fits both a baseline OLS model and a constrained regression that equalizes mean residuals across the specified groups, and saves summaries under `results/`.
