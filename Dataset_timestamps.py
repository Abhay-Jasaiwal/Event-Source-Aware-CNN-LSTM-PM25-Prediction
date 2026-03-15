import pandas as pd

files = [
    "Dwarka_Dataset_cleaned.csv",
    "ITO_Dataset_cleaned.csv",
    "PunjabiBagh_Dataset_cleaned.csv",
    "RKPuram_Dataset_cleaned.csv",
    "Rohini_Dataset_cleaned.csv"
]

for file in files:

    print(f"Processing {file}")

    df = pd.read_csv(file)

    df["datetimeLocal"] = pd.to_datetime(df["datetimeLocal"])

    df = df.set_index("datetimeLocal")

    # Create full hourly timeline
    df = df.resample("1H").mean()

    # Fill missing values
    df = df.interpolate(method="time")

    df = df.reset_index()

    new_name = file.replace("_cleaned.csv", "_final.csv")

    df.to_csv(new_name, index=False)

    print(f"Saved: {new_name}")

print("Hourly datasets completed.")