def predict_model(model,encoder, x_test, y_test):
    # Predictions
    predictions = model.predict(x_test)
    print(predictions)

    # Convert prediction to labels
    labels = encoder.inverse_transform(predictions)
    print(labels)

    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test,predictions)
    print("Accuracy:", accuracy)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,predictions,labels=[0,1,2])
    print(cm)

    # Classification report
    from sklearn.metrics import classification_report
    print(classification_report(y_test,predictions))

    return predictions,accuracy,cm