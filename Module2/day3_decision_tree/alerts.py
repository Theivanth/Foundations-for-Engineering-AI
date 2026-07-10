def alert_system(model,encoder):
    # Predict new data
    new_data = [[82,7,38,1950]]
    prediction = model.predict(new_data)

    result = encoder.inverse_transform(prediction)
    print(result)

    # Automatic Alert system
    if result[0] == "Critical":
        print("Critical Machine Failure")
    elif result[0] == "Warning":
        print("Maintenance required")
    else:
        print("Machine healthy")

    return result