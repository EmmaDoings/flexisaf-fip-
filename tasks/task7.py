from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_classification_model() -> None:
    """Train a supervised classification model and evaluate its predictions."""
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("Technique 1: Classification")
    print("Goal: predict whether a breast cancer sample is malignant or benign.")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=data.target_names))


def train_clustering_model() -> None:
    """Train an unsupervised clustering model and evaluate cluster separation."""
    data = load_iris()
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clusterer", KMeans(n_clusters=3, random_state=42, n_init=10)),
        ]
    )

    clusters = model.fit_predict(data.data)
    scaled_features = model.named_steps["scaler"].transform(data.data)

    print("\nTechnique 2: Clustering")
    print("Goal: group iris flowers by feature similarity without using labels.")
    print(f"Silhouette score: {silhouette_score(scaled_features, clusters):.3f}")
    print("Cluster sizes:")
    for cluster_id in sorted(set(clusters)):
        print(f"Cluster {cluster_id}: {(clusters == cluster_id).sum()} samples")


def main() -> None:
    print("Advanced Machine Learning Techniques Demonstration")
    print("Source learning path: Microsoft Learn - Create machine learning models\n")
    train_classification_model()
    train_clustering_model()


if __name__ == "__main__":
    main()
