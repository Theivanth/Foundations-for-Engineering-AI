from preprocessing import data
from visualization import plt_temperature, plt_aqi, plt_humidity
from prediction import future_aqi_prediction
from alerts import get_latest_aqi_alert, latest_aqi_value


def main():
    print("Welcome to the Environmental Data Analysis and Prediction System!")

    latest_aqi = latest_aqi_value()
    print(f"Latest AQI: {latest_aqi}")
    print(get_latest_aqi_alert(latest_aqi))

    sample_temperature = 33
    sample_humidity = 64
    predicted_aqi = future_aqi_prediction(sample_temperature, sample_humidity)
    print(f"Predicted Future AQI for temperature={sample_temperature}°C and humidity={sample_humidity}%: {predicted_aqi:.1f}")

    plt_temperature(data)
    plt_aqi(data)
    plt_humidity(data)


if __name__ == "__main__":
    main()

