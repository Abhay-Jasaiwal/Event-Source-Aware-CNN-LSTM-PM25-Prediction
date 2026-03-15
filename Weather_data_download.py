import requests
import pandas as pd

# NASA POWER API endpoint
url = "https://power.larc.nasa.gov/api/temporal/hourly/point"

params = {
    "parameters": "T2M,WS2M,RH2M,PRECTOTCORR",
    "community": "RE",
    "longitude": 77.20,
    "latitude": 28.61,
    "start": 20210101,
    "end": 20241231,
    "format": "JSON"
}

print("Downloading weather data from NASA POWER...")

response = requests.get(url, params=params)
data = response.json()

records = []

weather = data["properties"]["parameter"]

times = weather["T2M"].keys()

for t in times:
    records.append({
        "datetime": t,
        "T2M": weather["T2M"][t],
        "WS2M": weather["WS2M"][t],
        "RH2M": weather["RH2M"][t],
        "PRECTOTCORR": weather["PRECTOTCORR"][t]
    })

df = pd.DataFrame(records)

df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H")

df.to_csv("Delhi_weather.csv", index=False)

print("Weather dataset saved as Delhi_weather.csv")