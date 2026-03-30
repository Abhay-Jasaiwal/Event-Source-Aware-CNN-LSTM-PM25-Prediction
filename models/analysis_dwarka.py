import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/Dwarka_Final_2019_2022.csv")

# Target
target = "PM2.5"

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetimeLocal"], errors="coerce")
df = df.dropna(subset=["datetime"])
df = df.sort_values("datetime")

# Filter only 2020–2021
df = df[(df["datetime"].dt.year >= 2020) & (df["datetime"].dt.year <= 2021)]

# Extract year
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year

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

# Event features (best for scatter)
event_features = ["crop_burning", "winter_inversion", "diwali"]

# Clean
df = df.dropna()

# ------------------------------------------
# 📊 SCATTER PLOTS (TIME vs PM2.5)
# ------------------------------------------

for feature in event_features:

    plt.figure(figsize=(12,6))

    # Scatter with color showing feature intensity
    scatter = plt.scatter(
        df["datetime"],
        df["PM2.5"],
        c=df[feature],
        cmap="coolwarm",
        alpha=0.6
    )

    plt.colorbar(scatter, label=feature)

    plt.title(f"PM2.5 over Time (colored by {feature})")
    plt.xlabel("Time (2019–2022)")
    plt.ylabel("PM2.5")

    plt.show()

# ------------------------------------------
# 📊 MONTHLY SCATTER (PER FEATURE)
# ------------------------------------------

for feature in event_features:

    plt.figure(figsize=(12,6))

    sns.scatterplot(
        data=df,
        x="month",
        y=target,
        hue=feature,              # event impact
        style="year",             # 2020 vs 2021
        palette=["blue", "red"],
        alpha=0.6
    )

    plt.title(f"Monthly PM2.5 Pattern (2020–2021) - {feature}")
    plt.xlabel("Month")
    plt.ylabel("PM2.5")

    plt.legend(title=f"{feature} (0=No, 1=Yes)")
    plt.show()

# -----------------------------
# 📈 2. Correlation heatmap (VERY IMPORTANT)
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation with PM2.5")
plt.show()

# -----------------------------
# 📉 Dual-axis Line plots
# -----------------------------

for feature in features:

    fig, ax1 = plt.subplots(figsize=(12,5))

    # PM2.5 (left axis)
    ax1.plot(df["datetime"], df[target], color="blue", label="PM2.5")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("PM2.5", color="blue")
    ax1.tick_params(axis='y', labelcolor='blue')

    # Feature (right axis)
    ax2 = ax1.twinx()
    ax2.plot(df["datetime"], df[feature], color="red", linestyle="--", label=feature)
    ax2.set_ylabel(feature, color="red")
    ax2.tick_params(axis='y', labelcolor='red')

    # Title
    plt.title(f"PM2.5 vs {feature} (Time Series)")

    # Rotate x-axis
    fig.autofmt_xdate()

    plt.show()

    # ------------------------------------------
# 📊 BOXPLOTS (BEST FOR RESEARCH PAPER)
# ------------------------------------------

for feature in event_features:

    plt.figure(figsize=(6,4))

    sns.boxplot(x=df[feature], y=df[target])

    plt.title(f"Impact of {feature} on PM2.5")
    plt.xlabel(f"{feature} (0 = No, 1 = Yes)")
    plt.ylabel("PM2.5")

    plt.show()