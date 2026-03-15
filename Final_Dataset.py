import pandas as pd

station_files = [
    "Dwarka_Dataset_features.csv",
    "Rohini_Dataset_features.csv"
]

# Load weather dataset
weather = pd.read_csv("Delhi_weather.csv")

weather["datetime"] = pd.to_datetime(weather["datetime"])

for file in station_files:

    print(f"Merging weather with {file}")

    df = pd.read_csv(file)

    # Convert datetime
    df["datetimeLocal"] = pd.to_datetime(df["datetimeLocal"])

    # Remove timezone from datetimeLocal
    df["datetimeLocal"] = df["datetimeLocal"].dt.tz_localize(None)

    # Merge datasets
    merged = pd.merge(
        df,
        weather,
        left_on="datetimeLocal",
        right_on="datetime",
        how="left"
    )

    merged.drop(columns=["datetime"], inplace=True)

    new_name = file.replace("_features.csv", "_model_ready.csv")

    merged.to_csv(new_name, index=False)

    print(f"Saved: {new_name}")

print("Weather merging completed.")