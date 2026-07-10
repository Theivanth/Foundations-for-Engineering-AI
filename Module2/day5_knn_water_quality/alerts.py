def water_message(label):

    messages = {

        "Clean":
        "CLEAN: The water is safe for consumption.",

        "Polluted":
        "POLLUTED: The water is slightly polluted. Caution is advised.",

        "Hazardous":
        "HAZARDOUS: The water is highly contaminated. Do not consume."

    }

    return messages[label]


def display_prediction(prediction, probability):

    print("=" * 40)

    print("Predicted Quality:", prediction)

    print(f"Confidence: {probability:.1%}")

    print()

    print(water_message(prediction))

    print("=" * 40)