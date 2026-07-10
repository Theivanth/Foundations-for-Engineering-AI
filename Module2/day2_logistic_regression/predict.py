import pandas as pd


def predict_aqi(model, encoder, temperature, humidity, windspeed):
    # Build a DataFrame with the same feature names used during training
    cols = ["Temperature", "Humidity", "WindSpeed"]
    new_df = pd.DataFrame([[temperature, humidity, windspeed]], columns=cols)

    prediction = model.predict(new_df)
    result = encoder.inverse_transform(prediction)

    return result[0]
