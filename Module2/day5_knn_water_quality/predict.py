import pandas as pd


def predict_sample(model, X_train, y_train):

    sample = pd.DataFrame({

        "pH":[5.8],

        "Turbidity":[8.9],

        "DissolvedOxygen":[3.3],

        "Conductivity":[820],

        "Temperature":[32],

        "Nitrate":[14.2]

    })

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample).max()

    scaled = model.named_steps["scale"].transform(sample)

    knn = model.named_steps["knn"]

    distances, indices = knn.kneighbors(scaled)

    neighbours = X_train.iloc[indices[0]].copy()

    neighbours["Quality"] = y_train.iloc[indices[0]].values

    neighbours["Distance"] = distances[0]

    return prediction, probability, neighbours