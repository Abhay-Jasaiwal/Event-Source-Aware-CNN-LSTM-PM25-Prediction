import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/Rohini_Final_2019_2022.csv")

# Target
target = "PM2.5"

# Correct feature names from your dataset
features = [
    "crop_burning",
    "winter_inversion",
    "diwali",
    "T2M",
    "WS2M",
    "RH2M",
    "PRECTOTCORR"
]

# Clean
df = df.dropna()

# -----------------------------
# 📊 1. Scatter plots (MOST IMPORTANT)
# -----------------------------
for feature in features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[feature], y=df[target])
    plt.title(f"PM2.5 vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("PM2.5")
    plt.show()

# -----------------------------
# 📈 2. Correlation heatmap (VERY IMPORTANT)
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation with PM2.5")
plt.show()

# -----------------------------
# 📉 3. Line plots (time behavior)
# -----------------------------
for feature in features:
    plt.figure(figsize=(10,4))
    plt.plot(df[target], label="PM2.5")
    plt.plot(df[feature], label=feature)
    plt.title(f"PM2.5 vs {feature} (Time Series)")
    plt.legend()
    plt.show()