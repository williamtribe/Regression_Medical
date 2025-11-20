"""Reproduce fairness-aware risk adjustment using CMS Synthetic Data.

This script downloads a CSV from the Kaggle CMS synthetic dataset and runs both
baseline OLS regression and a constrained regression that equalizes mean
residuals across protected groups. It is designed to map to the slide workflow
summarized in docs/reproduction_plan.md.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from kagglehub import KaggleDatasetAdapter, load_dataset
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise ImportError(
        "kagglehub is required; install with `pip install kagglehub[pandas-datasets]`"
    ) from exc


@dataclass
class FairnessConfig:
    outcome: str
    features: Sequence[str]
    group_columns: Sequence[str]
    l2_penalty: float = 1e-3


@dataclass
class RegressionResult:
    coefficients: pd.Series
    residuals: pd.Series
    design_columns: List[str]


def _dummy_encode(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """One-hot encode selected columns while preserving original index."""
    encoded = pd.get_dummies(df, columns=list(columns), drop_first=True, dtype=float)
    return encoded


def load_kaggle_csv(file_path: str, pandas_kwargs: dict | None = None) -> pd.DataFrame:
    pandas_kwargs = pandas_kwargs or {}
    df = load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "anikannal/cms-synthetic-data",
        file_path,
        pandas_kwargs=pandas_kwargs,
    )
    return df


def build_design_matrix(df: pd.DataFrame, config: FairnessConfig) -> tuple[pd.DataFrame, pd.Series]:
    missing = [col for col in [config.outcome, *config.features] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    y = df[config.outcome].astype(float)
    X = df[list(config.features)].copy()
    X = _dummy_encode(X, [col for col in config.features if df[col].dtype == "object" or str(df[col].dtype) == "category"])
    X = sm.add_constant(X, has_constant="add")
    return X, y


def fit_baseline(df: pd.DataFrame, config: FairnessConfig) -> RegressionResult:
    X, y = build_design_matrix(df, config)
    model = sm.OLS(y, X).fit()
    residuals = y - model.predict(X)
    return RegressionResult(coefficients=model.params, residuals=residuals, design_columns=X.columns.tolist())


def fit_constrained(df: pd.DataFrame, config: FairnessConfig) -> RegressionResult:
    X, y = build_design_matrix(df, config)
    group_cols = [col for col in config.group_columns if col in df.columns]

    X_mat = X.to_numpy()
    y_vec = y.to_numpy()
    beta = cp.Variable(X_mat.shape[1])

    residuals = y_vec - X_mat @ beta
    objective = cp.Minimize(cp.sum_squares(residuals) + config.l2_penalty * cp.sum_squares(beta))

    constraints = []
    for group in group_cols:
        for level, idx in df.groupby(group).groups.items():
            if len(idx) == 0:
                continue
            group_mask = np.zeros(len(df))
            group_mask[list(idx)] = 1.0
            constraints.append(group_mask @ residuals == 0)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Constrained regression failed with status {problem.status}")

    beta_value = np.array(beta.value).flatten()
    residuals_series = pd.Series(y_vec - X_mat @ beta_value, index=df.index)
    coefficients = pd.Series(beta_value, index=X.columns)
    return RegressionResult(coefficients=coefficients, residuals=residuals_series, design_columns=X.columns.tolist())


def summarize_residuals(residuals: pd.Series, group_cols: Sequence[str], df: pd.DataFrame) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame({"mean_residual": [residuals.mean()]})

    valid_cols = [col for col in group_cols if col in df.columns]
    if not valid_cols:
        return pd.DataFrame({"mean_residual": [residuals.mean()]})

    grouped = df[valid_cols].copy()
    grouped["residual"] = residuals
    summary = grouped.groupby(valid_cols)["residual"].mean().reset_index().rename(columns={"residual": "mean_residual"})
    return summary


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_pipeline(args: argparse.Namespace) -> None:
    df = load_kaggle_csv(args.file_path, pandas_kwargs={"low_memory": False})

    config = FairnessConfig(
        outcome=args.outcome,
        features=args.features,
        group_columns=args.group_columns,
        l2_penalty=args.l2_penalty,
    )

    baseline = fit_baseline(df, config)
    constrained = fit_constrained(df, config)

    baseline_summary = summarize_residuals(baseline.residuals, config.group_columns, df)
    constrained_summary = summarize_residuals(constrained.residuals, config.group_columns, df)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary.to_csv(results_dir / "baseline_residuals.csv", index=False)
    constrained_summary.to_csv(results_dir / "constrained_residuals.csv", index=False)

    save_json(
        {
            "coefficients": baseline.coefficients.to_dict(),
            "design_columns": baseline.design_columns,
        },
        results_dir / "baseline_model.json",
    )

    save_json(
        {
            "coefficients": constrained.coefficients.to_dict(),
            "design_columns": constrained.design_columns,
            "status": "optimal",
        },
        results_dir / "constrained_model.json",
    )

    print("Baseline residual means (first 5):\n", baseline_summary.head())
    print("Constrained residual means (first 5):\n", constrained_summary.head())
    print(f"Saved outputs to {results_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fairness-aware risk adjustment reproduction")
    parser.add_argument("file_path", help="Path within Kaggle dataset to load (e.g., Beneficiary_Summary_File.csv)")
    parser.add_argument(
        "--outcome",
        default="TOTAL_SPENDING",
        help="Outcome column representing annual spending",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["AGE", "SEX", "RACE", "HCC_SCORE", "DUAL_STATUS", "DISABILITY_STATUS"],
        help="Feature columns for regression",
    )
    parser.add_argument(
        "--group-columns",
        dest="group_columns",
        nargs="+",
        default=["SEX", "RACE"],
        help="Columns to equalize residuals across",
    )
    parser.add_argument(
        "--l2-penalty",
        dest="l2_penalty",
        type=float,
        default=1e-3,
        help="L2 penalty strength for constrained regression",
    )
    parser.add_argument(
        "--results-dir",
        dest="results_dir",
        default="results",
        help="Directory to save summaries",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
