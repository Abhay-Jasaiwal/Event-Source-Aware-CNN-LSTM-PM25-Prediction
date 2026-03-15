import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

files = [
    "Dwarka_Dataset_model_ready.csv",
    "Rohini_Dataset_model_ready.csv"
]

TIME_STEPS = 24

for file in files:

    print(f"\nPreparing dataset: {file}")

    df = pd.read_csv(file)

    # Convert datetime
    df["datetimeLocal"] = pd.to_datetime(df["datetimeLocal"])

    # Sort data
    df = df.sort_values("datetimeLocal")

    # Remove datetime column for modelling
    data = df.drop(columns=["datetimeLocal"])

    # -------- FIX MISSING VALUES --------
    data = data.replace([np.inf, -np.inf], np.nan)

    # Interpolate missing values
    data = data.interpolate(method="linear")

    # Fill remaining edges
    data = data.fillna(method="bfill").fillna(method="ffill")

    # Final safety check
    if data.isnull().sum().sum() > 0:
        print("Still NaNs found — dropping rows")
        data = data.dropna()

    print("Remaining NaNs:", data.isnull().sum().sum())

    # -------- NORMALIZATION --------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    # Target variable index
    target_col = scaled_df.columns.get_loc("PM2.5")

    X = []
    y = []

    # -------- CREATE TIME WINDOWS --------
    for i in range(TIME_STEPS, len(scaled_df)):
        X.append(scaled_df.iloc[i-TIME_STEPS:i].values)
        y.append(scaled_df.iloc[i, target_col])

    X = np.array(X)
    y = np.array(y)

    print("Input shape:", X.shape)
    print("Target shape:", y.shape)

    # -------- TRAIN TEST SPLIT --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # Save arrays
    np.save(file.replace(".csv","_X_train.npy"), X_train)
    np.save(file.replace(".csv","_X_test.npy"), X_test)
    np.save(file.replace(".csv","_y_train.npy"), y_train)
    np.save(file.replace(".csv","_y_test.npy"), y_test)

    print(f"Dataset ready: {file}")

print("\nAll datasets successfully repaired and prepared.")