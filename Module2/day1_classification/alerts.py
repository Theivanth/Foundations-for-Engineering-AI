def generate_alerts(df):
    """
    Print alerts for dangerous AQI values.
    """

    for index, row in df.iterrows():

        if row["Status"] == "Dangerous":
            event_date = row["date"] if "date" in row else row.get("Day", "Unknown")
            location = row["location"] if "location" in row else "Unknown location"

            print(
                f"Alert: Dangerous AQI detected on {event_date} at {location} with AQI {row['AQI']}"
            )


def calculate_metrics(df):

    total_days = len(df)

    max_aqi = df["AQI"].max()

    max_row = df[df["AQI"] == max_aqi].iloc[0]

    safe_days = len(df[df["AQI"] <= 100])

    moderate_days = len(df[(df["AQI"] > 100) & (df["AQI"] <= 150)])

    dangerous_days = len(df[df["AQI"] > 150])

    return (
        total_days,
        max_aqi,
        max_row,
        safe_days,
        moderate_days,
        dangerous_days
    )


def generate_report(df):

    (
        total_days,
        max_aqi,
        max_row,
        safe_days,
        moderate_days,
        dangerous_days
    ) = calculate_metrics(df)

    report = f"""
================ AQI REPORT ================

Total Days : {total_days}

Maximum AQI : {max_aqi}

Location : {max_row['location']}

Safe Days : {safe_days}

Moderate Days : {moderate_days}

Dangerous Days : {dangerous_days}

============================================
"""

    with open("report.txt", "w") as f:
        f.write(report)

    print("report.txt generated successfully.")