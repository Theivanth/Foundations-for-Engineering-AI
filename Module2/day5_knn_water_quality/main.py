from train_knn import (
    load_data,
    split_data,
    train_model
)

from evaluate import (
    evaluate_model,
    tune_k
)

from predict import predict_sample

from alerts import display_prediction


def main():

    df = load_data("water_quality.csv")

    print(df.head())

    print(df.describe())

    print(df["Quality"].value_counts())

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    print("Training completed")

    evaluate_model(
        model,
        X_test,
        y_test
    )

    tune_k(
        X_train,
        y_train
    )

    prediction, probability, neighbours = predict_sample(
        model,
        X_train,
        y_train
    )

    display_prediction(
        prediction,
        probability
    )

    print()

    print("Nearest Samples")

    print(neighbours.sort_values("Distance"))


if __name__ == "__main__":

    main()