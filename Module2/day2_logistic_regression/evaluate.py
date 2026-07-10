from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

def evaluate_model(model, encoder, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)

    all_labels = list(range(len(encoder.classes_)))
    class_names = list(encoder.classes_)

    report = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=class_names
    )

    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    print("\nConfusion Matrix:")
    print(cm)

    pred_labels = encoder.inverse_transform(y_pred)
    print("\nPredicted Labels:")
    print(pred_labels)