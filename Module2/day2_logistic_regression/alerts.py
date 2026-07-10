def pollution_alert(category):
    if category == "Safe":
        print("Air quality is safe.")
    elif category == "Moderate":
        print("Air quality is moderate.")
    else:
        print("Air quality is dangerous.")

import pickle

def save_model(model, filename="model.pkl"):
    pickle.dump(model, open(filename, "wb"))
    print("Model saved.")


def load_model(filename="model.pkl"):
    return pickle.load(open(filename, "rb"))