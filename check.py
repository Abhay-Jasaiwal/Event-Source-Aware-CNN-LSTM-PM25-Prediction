import pandas as pd

df = pd.read_csv("Dwarka_Dataset_model_ready.csv")

# Convert datetime column
df["datetimeLocal"] = pd.to_datetime(df["datetimeLocal"])

# Check time gaps
print(df["datetimeLocal"].diff().value_counts().head())