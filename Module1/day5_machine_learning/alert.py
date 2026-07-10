def pollution_alert(df):
    latest_aqi = df["AQI"].iloc[-1]
    print(f"Latest AQI: {latest_aqi}")
    if latest_aqi > 150:
        print("Alert: Air Quality is Unhealthy! Consider staying indoors and using air purifiers.")
    elif latest_aqi > 100:
        print("Warning: Air Quality is Unhealthy for Sensitive Groups. People with respiratory issues should take precautions.")
    else:
        print("Good AQI: Air Quality is Good. Enjoy your day outdoors!")
    