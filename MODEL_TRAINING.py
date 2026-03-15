import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# We are loading the dataset here
X_train = np.load("Dwarka_Dataset_model_ready_X_train.npy")
X_test = np.load("Dwarka_Dataset_model_ready_X_test.npy")

y_train = np.load("Dwarka_Dataset_model_ready_y_train.npy")
y_test = np.load("Dwarka_Dataset_model_ready_y_test.npy")

print("Training shape:", X_train.shape)



early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# Applying LSTM
def build_lstm():

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),

        LSTM(64, return_sequences=True),
        LSTM(32),

        Dropout(0.3),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=Adam(0.001), loss="mse")

    return model


# Applying CNN
def build_cnn():

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),

        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),

        Conv1D(32, 3, activation="relu"),

        Flatten(),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=Adam(0.001), loss="mse")

    return model


# CNN-LSTM Hybrid
def build_cnn_lstm():

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),

        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),

        Conv1D(32, 3, activation="relu"),

        LSTM(64),

        Dropout(0.3),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=Adam(0.0005), loss="mse")

    return model


# Training + Evaluation
def train_and_evaluate(model, name):

    print(f"\nTraining {name} model")

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # Predictions
    y_pred = np.squeeze(model.predict(X_test))

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nResults for", name)
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)

    return history, y_pred, rmse, mae, r2


# Training the model here :) in both cnn and lstm
lstm_model = build_lstm()
cnn_model = build_cnn()
cnn_lstm_model = build_cnn_lstm()

hist_lstm, pred_lstm, rmse_lstm, mae_lstm, r2_lstm = train_and_evaluate(lstm_model, "LSTM")
hist_cnn, pred_cnn, rmse_cnn, mae_cnn, r2_cnn = train_and_evaluate(cnn_model, "CNN")
hist_cnn_lstm, pred_cnn_lstm, rmse_cnn_lstm, mae_cnn_lstm, r2_cnn_lstm = train_and_evaluate(cnn_lstm_model, "CNN-LSTM")


# Comparision
print("\nModel Comparison")

print("LSTM      RMSE:", rmse_lstm, "MAE:", mae_lstm, "R2:", r2_lstm)
print("CNN       RMSE:", rmse_cnn, "MAE:", mae_cnn, "R2:", r2_cnn)
print("CNN-LSTM  RMSE:", rmse_cnn_lstm, "MAE:", mae_cnn_lstm, "R2:", r2_cnn_lstm)


# plotting curves

plt.figure(figsize=(8,5))
plt.plot(hist_cnn_lstm.history["loss"], label="Train Loss")
plt.plot(hist_cnn_lstm.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss (CNN-LSTM)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Observing Actual vs Predictive
plt.figure(figsize=(10,5))
plt.plot(y_test[:200], label="Actual PM2.5")
plt.plot(pred_cnn_lstm[:200], label="Predicted PM2.5")
plt.legend()
plt.title("Actual vs Predicted PM2.5")
plt.show()


# Error Distribution

errors = y_test - pred_cnn_lstm

plt.figure(figsize=(6,4))
plt.hist(errors, bins=30)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()





cnn_lstm_model.save("cnn_lstm_air_quality_model.h5")