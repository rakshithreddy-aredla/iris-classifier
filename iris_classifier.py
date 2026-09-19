"""
Iris Flower Classifier - Multi-class Classification
====================================================
Classifies iris flowers into 3 species (Setosa, Versicolor, Virginica)
based on sepal/petal measurements.

Goes beyond a single model: compares 5 algorithms and visualizes the data
with matplotlib. The Iris dataset is the classic "hello world" of ML.

Libraries: scikit-learn, pandas, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model zoo - we'll compare several algorithms
MODELS = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}


def load_data():
    """Load the built-in Iris dataset."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = [iris.target_names[i] for i in iris.target]
    return df, iris


def compare_models(X, y):
    """Train and cross-validate every model, return ranked results."""
    results = {}
    for name, model in MODELS.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        results[name] = {
            "mean": scores.mean(),
            "std": scores.std(),
        }
        print(f"{name:<25} accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return results


def plot_decision_boundary(model, X, y, iris):
    """Visualize the decision boundary using sepal length vs petal length."""
    # Use only 2 features for 2D visualization
    feat1, feat2 = 0, 2  # sepal length, petal length
    X2 = X[:, [feat1, feat2]]
    model.fit(X2, y)

    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, edgecolor="k", cmap=plt.cm.RdYlBu)
    plt.xlabel(iris.feature_names[feat1])
    plt.ylabel(iris.feature_names[feat2])
    plt.title("Random Forest Decision Boundary (2 features)")
    plt.colorbar(scatter)
    plt.savefig("decision_boundary.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\nSaved visualization to decision_boundary.png")


def main():
    df, iris = load_data()
    print("Dataset shape:", df.shape)
    print("Classes:", df["species"].unique())
    print(df["species"].value_counts())

    X = df[iris.feature_names].values
    y = iris.target  # 0, 1, 2 for the three species

    # Compare all models with cross-validation
    print("\n=== Model Comparison (5-fold cross-validation) ===")
    results = compare_models(X, y)

    # Pick the best model and do a proper train/test evaluation
    best_name = max(results, key=lambda k: results[k]["mean"])
    best_model = MODELS[best_name]
    print(f"\nBest model: {best_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\n=== Final Evaluation ===")
    print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Predict a brand-new flower
    print("\n=== New Flower Prediction ===")
    new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # classic Setosa measurements
    pred = best_model.predict(new_flower)
    print("Predicted species:", iris.target_names[pred[0]])

    # Visualize
    plot_decision_boundary(best_model, X, y, iris)


if __name__ == "__main__":
    main()
