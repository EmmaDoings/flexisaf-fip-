import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def demo_basic_plots() -> None:
    # Simple synthetic dataset
    np.random.seed(42)
    x = np.arange(1, 21)
    y = 0.5 * x + np.random.normal(0, 3, size=len(x))

    # Line plot
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o")
    plt.title("Basic Line Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Scatter plot
    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, c=y, cmap="viridis", edgecolor="k")
    plt.title("Basic Scatter Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Y")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def demo_seaborn_hist_box() -> None:
    # Visualize with seaborn: histograms & box plots
    # We avoid downloading external files; use iris dataset from seaborn if available.
    try:
        import seaborn as sns

        df = sns.load_dataset("iris")
    except Exception:
        # Fallback dataset
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "feature1": rng.normal(0, 1, 200),
                "feature2": rng.normal(2, 1.5, 200),
                "target": rng.choice(["A", "B"], 200),
            }
        )

    # Histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        plt.hist(df[numeric_cols[0]].dropna().values, bins=20, color="#4c72b0", alpha=0.8)
        plt.title(f"Histogram: {numeric_cols[0]}")
        plt.xlabel(numeric_cols[0])
        plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.subplot(1, 2, 2)
    if numeric_cols and len(numeric_cols) > 1:
        plt.hist(df[numeric_cols[1]].dropna().values, bins=20, color="#dd8452", alpha=0.8)
        plt.title(f"Histogram: {numeric_cols[1]}")
        plt.xlabel(numeric_cols[1])
        plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Box plots (use seaborn if present)
    try:
        import seaborn as sns

        plt.figure(figsize=(12, 5))
        if numeric_cols:
            # If iris-like dataset exists, plot all numeric features in one boxplot
            # Melt for seaborn
            plot_df = df.melt(value_vars=numeric_cols, var_name="feature", value_name="value")
            sns.boxplot(x="feature", y="value", data=plot_df)
            plt.title("Box Plot (seaborn)")
            plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    except Exception:
        # Fallback box plot with matplotlib
        if numeric_cols:
            plt.figure(figsize=(12, 5))
            plt.boxplot([df[col].dropna().values for col in numeric_cols], labels=numeric_cols)
            plt.title("Box Plot")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()


def main() -> None:
    print("Launching Task 4 plots...")
    demo_basic_plots()
    demo_seaborn_hist_box()
    print("Done. Close plot windows to finish.")


if __name__ == "__main__":
    main()

