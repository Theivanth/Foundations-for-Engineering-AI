from pathlib import Path
import pandas as pd

from train import train_model
from evaluate import evaluate_model
from predict import predict_aqi
from alerts import pollution_alert
from plots import plot_aqi_trends

DATA_FILE = Path(__file__).resolve().parent / "pollution_data.csv"
df = pd.read_csv(DATA_FILE)

model, encoder, X_test, y_test, df = train_model()

evaluate_model(model, encoder, X_test, y_test,)

result = predict_aqi(model, encoder, 20, 45, 1)

pollution_alert(result)

# call plotting function
plot_aqi_trends(df)

# Save the model
from alerts import save_model
save_model(model)