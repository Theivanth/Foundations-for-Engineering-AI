from train_tree import train_model
from predict import predict_model
from visualize_tree import visualize
from alerts import alert_system
from model import save_and_analyze_model

def main():
    print("========== TRAINING MODEL ==========")
    trained_model, encoder, x_test, y_test, df = train_model()

    print("\n========== EVALUATING MODEL ==========")
    predictions, accuracy, cm = predict_model(
        trained_model,
        encoder,
        x_test,
        y_test
    )

    print("\n========== SAVING MODEL ==========")
    loaded_model, critical = save_and_analyze_model(
        trained_model,
        df
    )

    print("\n========== VISUALIZATION ==========")
    visualize()

    print("\n========== ALERT SYSTEM ==========")
    alert_result = alert_system(
        loaded_model,
        encoder
    )

    print("\n========== PROCESS COMPLETE ==========")
    print("Final Alert Status:", alert_result[0])


if __name__ == "__main__":
    main()


