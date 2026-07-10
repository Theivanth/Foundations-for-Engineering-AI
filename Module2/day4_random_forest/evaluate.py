import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.3f}")

    print(confusion_matrix(y_test, predictions))

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    return predictions


def plot_feature_importance(model, feature_names):

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    importance.sort_values().plot(
        kind="barh",
        title="Random Forest Feature Importance"
    )

    plt.show()