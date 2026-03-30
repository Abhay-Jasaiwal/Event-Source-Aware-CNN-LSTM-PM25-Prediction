from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from utils.preprocessing import load_and_preprocess
from utils.metrics import evaluate
from utils.plots import plot_loss, plot_prediction


X_train, X_test, y_train, y_test = load_and_preprocess("data/Rohini_Final_2019_2022.csv", "PM2.5")

print("Shape:", X_train.shape)

early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    Conv1D(64, 3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(1),

    Conv1D(32, 1, activation="relu"),
    BatchNormalization(),

    LSTM(64),

    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer=Adam(0.0005), loss="mse")

history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[early_stop])

y_pred = model.predict(X_test).flatten()

rmse, mae, r2 = evaluate(y_test, y_pred)

print("\nCNN-LSTM Rohini")
print("RMSE:", rmse, "MAE:", mae, "R2:", r2)

plot_loss(history, "CNN-LSTM Loss (Rohini)")
plot_prediction(y_test, y_pred, "CNN-LSTM Prediction (Rohini)")

# Return model + test data
def get_model_and_data():
    return model, X_test, y_test