import pandas as pd


def predict_machine(model):

    new_machine = pd.DataFrame([{
        "Temperature": 50,
        "Vibration": 4,
        "Pressure": 40,
        "RPM": 2030,
        "Current": 6,
        "OperatingHours": 900
    }])

    prediction = model.predict(new_machine)[0]

    probabilities = model.predict_proba(new_machine)[0]

    return prediction, probabilities