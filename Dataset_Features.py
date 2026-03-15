import pandas as pd

files = [
    "Dwarka_Dataset_final.csv",
    "Rohini_Dataset_final.csv"
]

diwali_dates = [
    "2021-11-04",
    "2022-10-24",
    "2023-11-12",
    "2024-11-01"
]

for file in files:

    print(f"Processing {file}")

    df = pd.read_csv(file)

    # Convert datetime
    df["datetimeLocal"] = pd.to_datetime(df["datetimeLocal"])

    # ---- TIME FEATURES ----
    df["hour"] = df["datetimeLocal"].dt.hour
    df["day"] = df["datetimeLocal"].dt.day
    df["month"] = df["datetimeLocal"].dt.month
    df["day_of_week"] = df["datetimeLocal"].dt.dayofweek

    # ---- EVENT FEATURES ----

    # Crop burning (Oct–Nov)
    df["crop_burning"] = df["month"].isin([10,11]).astype(int)

    # Winter inversion (Dec–Jan)
    df["winter_inversion"] = df["month"].isin([12,1]).astype(int)

    # Diwali indicator
    df["diwali"] = df["datetimeLocal"].dt.date.astype(str).isin(diwali_dates).astype(int)

    # Sort by time
    df = df.sort_values("datetimeLocal")

    # Save new dataset
    new_name = file.replace("_final.csv", "_features.csv")
    df.to_csv(new_name, index=False)

    print(f"Saved: {new_name}")

print("Feature engineering completed.")