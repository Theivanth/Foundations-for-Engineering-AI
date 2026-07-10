from train_forest import (
    FEATURES,
    load_data,
    validate_dataset,
    split_data,
    train_model,
    save_model
)

from evaluate import (
    evaluate_model,
    plot_feature_importance
)

from predict import predict_machine

from alerts import generate_alert


def main():

    df = load_data("maintenance_data.csv")

    validate_dataset(df)

    print("Dataset validation passed")

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    print("Training Complete")

    evaluate_model(
        model,
        X_test,
        y_test
    )

    plot_feature_importance(
        model,
        FEATURES
    )

    prediction, probabilities = predict_machine(model)

    generate_alert(
        prediction,
        probabilities,
        model.classes_
    )

    save_model(model)

    print("Model package saved.")


if __name__ == "__main__":
    main()