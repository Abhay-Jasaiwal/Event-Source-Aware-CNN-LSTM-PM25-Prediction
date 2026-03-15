import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


model = load_model("cnn_lstm_air_quality_model.h5")

print("Model loaded successfully")


X_test = np.load("Dwarka_Dataset_model_ready_X_test.npy")
y_test = np.load("Dwarka_Dataset_model_ready_y_test.npy")

print("Test data shape:", X_test.shape)



y_pred = model.predict(X_test)

# remove extra dimension
y_pred = np.squeeze(y_pred)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTesting Results")
print("RMSE:", rmse)
print("MAE :", mae)
print("R²  :", r2)


plt.figure(figsize=(10,5))
plt.plot(y_test[:200], label="Actual PM2.5")
plt.plot(y_pred[:200], label="Predicted PM2.5")

plt.title("Actual vs Predicted PM2.5")
plt.xlabel("Time Step")
plt.ylabel("PM2.5")

plt.legend()
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")

plt.title("Prediction Scatter Plot")

plt.show()