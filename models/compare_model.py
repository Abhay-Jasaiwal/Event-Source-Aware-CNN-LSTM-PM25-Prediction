import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.evaluation import evaluate_model

# Import models
from models.cnn_dwarka import get_model_and_data as cnn_dwarka
from models.lstm_dwarka import get_model_and_data as lstm_dwarka
from models.cnn_lstm_dwarka import get_model_and_data as cnn_lstm_dwarka

from models.cnn_rohini import get_model_and_data as cnn_rohini
from models.lstm_rohini import get_model_and_data as lstm_rohini
from models.cnn_lstm_rohini import get_model_and_data as cnn_lstm_rohini


results = []

# -----------------------------
# Dwarka
# -----------------------------
for name, func in [
    ("CNN", cnn_dwarka),
    ("LSTM", lstm_dwarka),
    ("CNN-LSTM", cnn_lstm_dwarka),
]:
    model, X_test, y_test = func()
    rmse, mae, r2 = evaluate_model(model, X_test, y_test)
    results.append([name, "Dwarka", rmse, mae, r2])


# -----------------------------
# Rohini
# -----------------------------
for name, func in [
    ("CNN", cnn_rohini),
    ("LSTM", lstm_rohini),
    ("CNN-LSTM", cnn_lstm_rohini),
]:
    model, X_test, y_test = func()
    rmse, mae, r2 = evaluate_model(model, X_test, y_test)
    results.append([name, "Rohini", rmse, mae, r2])


# DataFrame
df = pd.DataFrame(results, columns=["Model", "Dataset", "RMSE", "MAE", "R2"])

print(df)

# -----------------------------
# Graphs
# -----------------------------
plt.figure(figsize=(10,5))
sns.barplot(x="Model", y="RMSE", hue="Dataset", data=df)
plt.title("RMSE Comparison")
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x="Model", y="R2", hue="Dataset", data=df)
plt.title("R2 Comparison")
plt.show()