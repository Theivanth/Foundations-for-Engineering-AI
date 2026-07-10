import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))

    print("Accuracy:",
          accuracy_score(y_test, predictions))

    print(confusion_matrix(y_test, predictions))


def tune_k(X_train, y_train):

    max_k = min(9, len(X_train) - 1)
    k_values = [k for k in range(1, max_k + 1) if k % 2 == 1]

    scores = []

    for k in k_values:

        candidate = Pipeline(steps=[
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])

        result = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=4,
            scoring="f1_macro"
        )

        scores.append(result.mean())

    plt.plot(list(k_values), scores, marker="o")

    plt.xlabel("k")

    plt.ylabel("Mean F1")

    plt.title("Selecting Best k")

    plt.grid(True)

    plt.tight_layout()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "k_selection.png")
    plt.close()
    print(f"Saved k selection plot to {output_dir / 'k_selection.png'}")