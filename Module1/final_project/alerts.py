from preprocessing import data


def latest_aqi_value():
    return data['AQI'].iloc[-1]


def get_latest_aqi_alert(latest_aqi):
    if latest_aqi > 150:
        return "Alert: Air Quality Index is unhealthy!"
    elif latest_aqi > 100:
        return "Alert: Air Quality Index is moderate!"
    return "Air Quality Index is good."

