import numpy as np
import pandas as pd


def analyze_dataset(df: pd.DataFrame) -> None:
    print("=== 1) Explore the data ===")
    print("--- head() ---")
    print(df.head())

    print("\n--- info() ---")
    df.info()

    print("\n--- describe() ---")
    print(df.describe(include="all"))

    print("\n=== 2) Calculate basic statistics (numeric columns) ===")
    numeric_df = df.select_dtypes(include=[np.number])

    # Mean, median, std
    means = numeric_df.mean(numeric_only=True)
    medians = numeric_df.median(numeric_only=True)
    stds = numeric_df.std(numeric_only=True, ddof=1)

    # Mode: may be multiple; take the first mode value per column
    modes = {}
    for col in numeric_df.columns:
        mode_series = numeric_df[col].mode(dropna=True)
        modes[col] = mode_series.iloc[0] if len(mode_series) > 0 else np.nan
    modes = pd.Series(modes)

    stats_table = pd.DataFrame(
        {
            "mean": means,
            "median": medians,
            "mode": modes,
            "std_dev": stds,
        }
    )
    print("\n--- Per-column statistics ---")
    print(stats_table)

    # Correlation
    print("\n--- Correlation matrix (data.corr()) ---")
    corr = numeric_df.corr()
    print(corr)


def main() -> None:
    # Public small dataset (built-in via Seaborn; does not require extra downloads)
    # If seaborn is not available, you can replace this with any local CSV.
    try:
        import seaborn as sns

        df = sns.load_dataset("iris")
    except Exception:
        # Fallback: create a small example dataset.
        df = pd.DataFrame(
            {
                "sepal_length": [5.1, 4.9, 4.7, 4.6],
                "sepal_width": [3.5, 3.0, 3.2, 3.1],
                "petal_length": [1.4, 1.4, 1.3, 1.5],
                "petal_width": [0.2, 0.2, 0.2, 0.2],
                "species": ["setosa", "setosa", "setosa", "setosa"],
            }
        )

    analyze_dataset(df)


if __name__ == "__main__":
    main()

