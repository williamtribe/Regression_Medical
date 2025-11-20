# Fair Risk Adjustment Reproduction Plan

This guide summarizes how to recreate the core analyses from the JAMA Health Forum article on Medicare risk adjustment fairness using the CMS Synthetic Data set available on Kaggle. It is tailored to the provided presentation slides and focuses on constrained regression to reduce overpayment for overrepresented beneficiary groups.

## Data Access
1. Install the Kaggle CLI and authenticate (`pip install kaggle` and place your API token in `~/.kaggle/kaggle.json`).
2. Download the CMS Synthetic Data set:
   ```bash
   kaggle datasets download -d anikannal/cms-synthetic-data -p data
   unzip -d data data/cms-synthetic-data.zip
   ```
3. Inspect the files to confirm availability of demographics (age, sex, race/ethnicity), diagnostic codes, utilization, and spending variables similar to those used in the JAMA Health Forum study.

## Cohort Construction
- Filter to beneficiaries with complete demographics and spending outcomes for the target year.
- Derive outcome variables: annual total cost and utilization counts (inpatient, outpatient, ED visits) as in the study.
- Create subgroup indicators: age bands, sex, race/ethnicity, dual-eligibility, disability status.

## Baseline Risk Adjustment Model
- Fit a standard linear regression (or generalized linear model) predicting annual cost using demographics and clinical risk factors.
- Estimate group-level prediction errors (observed minus predicted spending) to identify over- or under-payment patterns.
- Summarize results in tables that map to the slide narrative about bias arising from disproportionate sample representation.

## Constrained Regression for Fairness
- Implement constraints to equalize average prediction error across protected groups (e.g., sex or race/ethnicity). In practice, this can be done with quadratic programming or Lagrangian reweighting that adds penalty terms for group-level residual imbalances.
- Compare baseline vs. constrained models on:
  - Mean prediction error by group.
  - Overall fit metrics (R², RMSE) to verify predictive performance is largely retained.
  - Visuals for slides: bar charts of group-level residuals before/after constraints and tables of payment differentials.

## Slide-Ready Talking Points
- **Fairness Motivation:** Traditional regression can encode bias when overrepresented groups drive parameter estimates; constrained regression explicitly corrects residual imbalances rather than assuming demographic differences are clinical differences.
- **Effect on Spending Predictions:** Constrained models reduce overpayment to advantaged groups while maintaining total predictive accuracy, mirroring the published findings.
- **Policy Interpretation:** Adjusting payment formulas this way improves equity without discarding clinical relevance—aligning with the study’s conclusion that constrained regression “benefits health conditions that are overrepresented among minority groups while maintaining overall fit.”

## Reproducible Analysis Structure
- Place notebooks or scripts under `analysis/` (e.g., `analysis/baseline_model.ipynb`, `analysis/constrained_model.ipynb`).
- Use a shared `environment.yml` or `requirements.txt` with `pandas`, `numpy`, `statsmodels`, `cvxpy`, and plotting libraries.
- Keep an outputs folder (`results/`) with CSV summaries and figures that map directly to the three provided slides.

## Validation Checklist
- [ ] Data downloaded and preprocessed with subgroup indicators.
- [ ] Baseline model fitted and group residual disparities computed.
- [ ] Constrained regression implemented and disparities reduced while preserving fit.
- [ ] Figures/tables produced for each slide’s key message.
