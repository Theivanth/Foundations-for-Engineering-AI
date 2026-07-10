from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
from preprocessing import data


def train_aqi_model():
    x = data[["Temperature_C", "Humidity_Percent"]]
    y = data["AQI"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    error = mean_absolute_error(y_test, predictions)
    return model, error


def future_aqi_prediction(temperature, humidity):
    input_data = pd.DataFrame([[temperature, humidity]], columns=["Temperature_C", "Humidity_Percent"])
    model, error = train_aqi_model()
    return model.predict(input_data)[0]


def correlation_analysis():
    return data.corr()


def feature_selection(x, y):
    return x, y


