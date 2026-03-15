import pandas as pd
import os

# List of dataset files
files = [
    "Dwarka_Dataset.csv",
    "ITO_Dataset.csv",
    "PunjabiBagh_Dataset.csv",
    "RKPuram_Dataset.csv",
    "Rohini_Dataset.csv"
]

for file in files:

    print(f"Processing {file}...")

    df = pd.read_csv(file)

    # Convert datetime
    df["datetimeLocal"] = pd.to_datetime(df["datetimeLocal"])

    # Fix date range (2021–2024)
    df = df[
        (df["datetimeLocal"] >= "2021-01-01") &
        (df["datetimeLocal"] < "2025-01-01")
    ]

    # Fix CO unit mismatch (ppb → µg/m³)
    mask = (df["parameter"] == "co") & (df["unit"] == "ppb")
    df.loc[mask, "value"] = df.loc[mask, "value"] * 1.145
    df.loc[mask, "unit"] = "µg/m³"

    # Pivot table (long → wide)
    pivot_df = df.pivot_table(
        index="datetimeLocal",
        columns="parameter",
        values="value"
    ).reset_index()

    # Rename columns
    pivot_df.rename(columns={
        "pm25": "PM2.5",
        "pm10": "PM10",
        "no2": "NO2",
        "so2": "SO2",
        "co": "CO",
        "o3": "O3"
    }, inplace=True)

    # Sort by datetime
    pivot_df = pivot_df.sort_values("datetimeLocal")

    # Save cleaned file
    new_name = file.replace(".csv", "_cleaned.csv")
    pivot_df.to_csv(new_name, index=False)

    print(f"Saved: {new_name}")

print("All datasets cleaned successfully.")