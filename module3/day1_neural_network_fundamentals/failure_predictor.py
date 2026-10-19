import torch
import numpy as np

class FailurePredictor:

    def __init__(self, model, class_names, scaler=None):
        self.model = model
        self.scaler = scaler
        self.class_names = class_names

        self.model.eval()

    def get_inputs(self):

        print("\nEnter sensor values")

        temperature = float(input("Temperature (°C): "))
        vibration = float(input("Vibration (mm/s): "))
        pressure = float(input("Pressure (kPa): "))
        rpm = float(input("RPM: "))
        current = float(input("Current (A): "))
        noise = float(input("Noise (dB): "))

        data = np.array([
            [
                temperature,
                vibration,
                pressure,
                rpm,
                current,
                noise
            ]
        ])

        return data

    def predict(self, data):

        # Apply scaling only when the model was trained with that scaler.
        input_data = self.scaler.transform(data) if self.scaler is not None else data

        input_tensor = torch.tensor(
            input_data,
            dtype=torch.float32
        )

        with torch.no_grad():

            output = self.model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_index = torch.argmax(
                probabilities,
                dim=1
            ).item()

        predicted_class = self.class_names[
            predicted_index
        ]

        return predicted_class, probabilities[0]

    def run(self):

        data = self.get_inputs()

        predicted_class, probabilities = self.predict(
            data
        )

        print("\n--- Prediction Result ---")

        print(
            "Predicted Condition:",
            predicted_class
        )

        print("\nProbabilities:")

        for name, probability in zip(
            self.class_names,
            probabilities
        ):

            print(
                f"{name}: "
                f"{probability.item()*100:.2f}%"
            )