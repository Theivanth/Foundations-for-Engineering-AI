def generate_alert(prediction, probabilities, classes):

    print("=" * 40)

    print("Prediction:", prediction)

    print()

    print("Prediction Confidence")

    for label, probability in zip(classes, probabilities):

        print(f"{label}: {probability:.2%}")

    print("=" * 40)

    if prediction.lower() == "critical":

        print("ALERT: Immediate maintenance required.")

    elif prediction.lower() == "warning":

        print("Maintenance should be scheduled soon.")

    else:

        print("Machine operating normally.")