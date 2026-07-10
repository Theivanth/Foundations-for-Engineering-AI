from preprocessing import preprocess_data
from visualization import plt_temperature, plt_aqi, correlation_analysis
from alert import pollution_alert
from prediction import train_model
import pandas as pd


def main():
    # Preprocess the data
    file_path = 'environment_data.csv'  # Path to your dataset
    
    #process the data
    processed_data, scalers = preprocess_data(file_path)

    # call visualization functions
    plt_temperature(processed_data)
    plt_aqi(processed_data)
    correlation_analysis(processed_data)

    # Call prediction function
    model = train_model(processed_data)

    # Call alert function
    pollution_alert(processed_data)

      
    # # Further processing or model training can be done here
    # print(processed_data.info())

    # futrure predition can be done here
    future_data = [30, 70]  # Example future data for prediction (Temperature, Humidity)

    # Scale future input using the same min/max used in preprocessing
    feature_order = ["Temperature", "Humidity"]
    scaled = []
    for i, col in enumerate(feature_order):
        if col in scalers:
            mn = scalers[col]['min']
            mx = scalers[col]['max']
            if mx - mn != 0:
                scaled.append((future_data[i] - mn) / (mx - mn))
            else:
                scaled.append(0.0)
        else:
            scaled.append(future_data[i])

    future_df = pd.DataFrame([scaled], columns=feature_order)
    future_prediction = model.predict(future_df)
    print(f"Predicted AQI for future data {[future_data]}: {future_prediction[0]}")
    

    

if __name__ == "__main__":    
    main()


