def save_and_analyze_model(trained_model,df):
    # Save model
    import pickle
    pickle.dump(trained_model, open("tree_model.pkl","wb"))

    # Load model
    loaded_model = pickle.load(open("tree_model.pkl","rb"))
    prediction1 = loaded_model.predict([[80,7,38,1900]])
    print(prediction1)

    critical = df[df["Status"] == 0]
    print(critical)

    return loaded_model,critical