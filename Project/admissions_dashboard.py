import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REQUIRED_COLUMNS = {"Admission_Status"}
NUMERIC_CANDIDATES = ["GPA", "SAT_Score", "Extracurricular_Activities"]


@dataclass
class LoadResult:
    df: pd.DataFrame
    warnings: List[str]


def _safe_read_csv(path: str) -> pd.DataFrame:
    # Keep engine defaults; rely on pandas for robust parsing.
    return pd.read_csv(path)


def load_and_clean(csv_path: str) -> LoadResult:
    warnings: List[str] = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    try:
        df = _safe_read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at '{csv_path}': {e}")

    # Basic schema validation
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required column(s): {missing_required}. Found: {list(df.columns)}")

    # Handle duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        warnings.append(f"Dropped {before - after} duplicate row(s).")

    # Coerce numeric columns if present
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean Admission_Status values
    # Normalize whitespace + capitalization
    df["Admission_Status"] = df["Admission_Status"].astype(str).str.strip()

    allowed_status = {"Accepted", "Rejected", "Waitlisted", "accepted", "rejected", "waitlisted"}
    unknown = ~df["Admission_Status"].isin(allowed_status)
    if unknown.any():
        bad_vals = sorted(df.loc[unknown, "Admission_Status"].unique().tolist())
        warnings.append(
            "Found unknown Admission_Status value(s); affected rows will be dropped: "
            + ", ".join(map(str, bad_vals[:10]))
        )
        df = df.loc[~unknown].copy()

    # Map to canonical capitalization
    status_map = {
        "accepted": "Accepted",
        "rejected": "Rejected",
        "waitlisted": "Waitlisted",
    }
    df["Admission_Status"] = df["Admission_Status"].replace(status_map)

    # Drop rows with missing critical values
    before = len(df)
    df = df.dropna(subset=["Admission_Status"])
    after = len(df)
    if after < before:
        warnings.append(f"Dropped {before - after} row(s) due to missing Admission_Status.")

    return LoadResult(df=df, warnings=warnings)


def compute_metrics(df: pd.DataFrame) -> Dict[str, object]:
    metrics: Dict[str, object] = {}

    total = len(df)
    counts = df["Admission_Status"].value_counts(dropna=False)
    accepted = int(counts.get("Accepted", 0))
    rejected = int(counts.get("Rejected", 0))
    waitlisted = int(counts.get("Waitlisted", 0))

    metrics["total_applicants"] = total
    metrics["accepted_count"] = accepted
    metrics["rejected_count"] = rejected
    metrics["waitlisted_count"] = waitlisted

    metrics["acceptance_rate"] = (accepted / total * 100.0) if total else 0.0
    metrics["waitlist_rate"] = (waitlisted / total * 100.0) if total else 0.0
    metrics["rejection_rate"] = (rejected / total * 100.0) if total else 0.0

    # Numeric summaries
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                metrics[f"{col}_mean"] = float(s.mean())
                metrics[f"{col}_median"] = float(s.median())
                metrics[f"{col}_min"] = float(s.min())
                metrics[f"{col}_max"] = float(s.max())
                metrics[f"{col}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0

    return metrics


def generate_insights(metrics: Dict[str, object], df: pd.DataFrame, warnings: List[str]) -> str:
    accepted = metrics.get("accepted_count", 0)
    total = metrics.get("total_applicants", 0)
    acceptance_rate = metrics.get("acceptance_rate", 0.0)
    counts = df["Admission_Status"].value_counts(dropna=False)
    top_outcome = counts.index[0] if len(counts) else None
    top_outcome_count = int(counts.iloc[0]) if len(counts) else 0

    lines: List[str] = []

    if total:
        lines.append(
            f"Acceptance rate is {acceptance_rate:.1f}% ({accepted}/{total} accepted)."
        )
    else:
        lines.append("Acceptance rate could not be computed (no valid rows).")

    if top_outcome is not None:
        share = (top_outcome_count / total * 100.0) if total else 0.0
        lines.append(
            f"The most common admission outcome is {top_outcome} with {top_outcome_count} applicants ({share:.1f}%)."
        )

    # Numeric highlights
    for col in NUMERIC_CANDIDATES:
        mean_key = f"{col}_mean"
        med_key = f"{col}_median"
        if mean_key in metrics and med_key in metrics:
            lines.append(
                f"Average {col.replace('_', ' ')} is {metrics[mean_key]:.2f} (median {metrics[med_key]:.2f})."
            )

    # Dataset feature availability
    if any(warnings):
        lines.append("Data notes: " + " ".join(warnings[:3]))

    # Gender/program requirements (template)
    # Current dataset does not contain these columns; we intentionally do not fabricate them.
    lines.append(
        "Note: This dataset does not include gender or program-preference columns, so gender ratios and top-program demand are skipped."
    )

    return " ".join(lines)


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def plot_outcome_distribution(df: pd.DataFrame, outdir: str) -> Tuple[str, str]:
    counts = df["Admission_Status"].value_counts()

    # Bar
    plt.figure(figsize=(7, 4.5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title("Admission Outcome Counts")
    plt.xlabel("Admission Status")
    plt.ylabel("Number of Applicants")
    plt.tight_layout()
    bar_path = os.path.join(outdir, "outcome_counts.png")
    plt.savefig(bar_path, dpi=160)
    plt.close()

    # Pie
    plt.figure(figsize=(6.5, 6.5))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Admission Outcome Share")
    plt.tight_layout()
    pie_path = os.path.join(outdir, "outcome_share_pie.png")
    plt.savefig(pie_path, dpi=160)
    plt.close()

    return bar_path, pie_path


def plot_histograms(df: pd.DataFrame, outdir: str) -> List[str]:
    paths: List[str] = []
    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    for col in numeric_cols:
        data = df[col].dropna()
        if not len(data):
            continue

        plt.figure(figsize=(7, 4.5))
        sns.histplot(data, kde=True, bins=15, color="#4c72b0")
        plt.title(f"Distribution of {col.replace('_', ' ')}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        p = os.path.join(outdir, f"hist_{col}.png")
        plt.savefig(p, dpi=160)
        plt.close()
        paths.append(p)

    return paths


def export_summary(metrics: Dict[str, object], insights: str, df: pd.DataFrame, outdir: str) -> None:
    summary_df = pd.DataFrame([metrics])
    summary_csv = os.path.join(outdir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    outcome_counts = df["Admission_Status"].value_counts().reset_index()
    outcome_counts.columns = ["Admission_Status", "Count"]

    # Render to HTML with a simple layout.
    insights_html = f"<p><strong>AI-powered narrative (rule-based):</strong> {insights}</p>"
    warn_html = ""  # warnings are expected to already be in the narrative

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Admissions Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f6f6f6; }}
    .muted {{ color: #666; }}
    img {{ max-width: 720px; margin: 14px 0; display:block; }}
  </style>
</head>
<body>
  <h1>Admissions EDA Summary</h1>

  {insights_html}
  {warn_html}

  <h2>Key Metrics</h2>
  {summary_df.to_html(index=False, classes='metrics', border=0)}

  <h2>Outcome Counts</h2>
  {outcome_counts.to_html(index=False, border=0)}

  <p class=\"muted\">Charts saved as PNG in the same outputs folder.</p>
</body>
</html>
"""

    summary_html = os.path.join(outdir, "summary.html")
    with open(summary_html, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Admissions EDA dashboard generator")
    parser.add_argument("--csv", required=True, help="Path to the input CSV")
    parser.add_argument("--outdir", default="Project/outputs", help="Output directory")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    load_result = load_and_clean(args.csv)
    df = load_result.df

    _ensure_outdir(args.outdir)

    metrics = compute_metrics(df)
    insights = generate_insights(metrics, df, load_result.warnings)

    # Plots
    plot_outcome_distribution(df, args.outdir)
    plot_histograms(df, args.outdir)

    # Export
    export_summary(metrics, insights, df, args.outdir)

    print("Done.")
    print(f"- Rows used: {len(df)}")
    print(f"- Acceptance rate: {metrics.get('acceptance_rate', 0.0):.1f}%")
    print(f"- Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()

