import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_and_evaluate_random_forest() -> None:
    # ==========================================
    # 1. DATA ACQUISITION & PREPARATION
    # ==========================================
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}\n")

    # ==========================================
    # TECHNIQUE 1: ENSEMBLE LEARNING (Random Forest)
    # ==========================================
    baseline_rf = RandomForestClassifier(random_state=42)
    baseline_rf.fit(X_train, y_train)

    baseline_preds = baseline_rf.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    print("--- Baseline Random Forest Classifier ---")
    print(f"Accuracy: {baseline_accuracy:.4f}\n")

    # ==========================================
    # TECHNIQUE 2: HYPERPARAMETER OPTIMIZATION (Grid Search)
    # ==========================================
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    print("Running Hyperparameter Optimization via Grid Search...")
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}\n")

    best_model = grid_search.best_estimator_
    optimized_preds = best_model.predict(X_test)
    optimized_accuracy = accuracy_score(y_test, optimized_preds)

    print("--- Optimized Random Forest Classifier ---")
    print(f"Accuracy: {optimized_accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, optimized_preds))

    # ==========================================
    # VISUALIZATIONS FOR YOUR DELIVERABLE
    # ==========================================
    # 1) Confusion matrix
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, optimized_preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=data.target_names,
        yticklabels=data.target_names,
    )
    plt.title("Confusion Matrix (Optimized Model)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # 2) Top feature importances
    plt.subplot(1, 2, 2)
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]  # top 10 features

    plt.barh(
        range(len(indices)),
        importances[indices],
        align="center",
        color="#4c72b0",
        alpha=0.9,
    )
    plt.yticks(range(len(indices)), [data.feature_names[i] for i in indices])
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Relative Importance")

    plt.tight_layout()
    plt.show()


def main() -> None:
    sns.set_theme(style="whitegrid")
    train_and_evaluate_random_forest()


if __name__ == "__main__":
    main()

