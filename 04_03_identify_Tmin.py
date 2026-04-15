import pandas as pd

file_2023 = r"2023/ERA5Land_2023_tmin_anomaly_0525_0907_by_diming.csv"
file_threshold = r"2023/threshold/ERA5Land_1981_2010_tmin_anomaly_threshold_p90_0601_0831_by_diming.csv"

df = pd.read_csv(file_2023)
df_threshold = pd.read_csv(file_threshold)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df_threshold["date"] = pd.to_datetime(df_threshold["date"], format="%m-%d", errors="coerce")

df["month_day"] = df["date"].dt.strftime("%m-%d")
df_threshold["month_day"] = df_threshold["date"].dt.strftime("%m-%d")

df = df.merge(
    df_threshold[["region_name", "month_day", "temp_min_anomaly_threshold"]],
    on=["region_name", "month_day"],
    how="left"
)

mask = (
    ((df["date"].dt.month == 6) & (df["date"].dt.day >= 1)) |
    (df["date"].dt.month == 7) |
    (df["date"].dt.month == 8) |
    ((df["date"].dt.month == 8) & (df["date"].dt.day <= 31))
)

df["day"] = 0
df["day_intensity"] = 0.0

cond = mask & (df["temp_min_anomaly"] > df["temp_min_anomaly_threshold"])

df.loc[cond, "day"] = 1
df.loc[cond, "day_intensity"] = df.loc[cond, "temp_min_anomaly"] - df.loc[cond, "temp_min_anomaly_threshold"]

df = df.drop(columns=["month_day"])

df.to_csv(file_2023, index=False, encoding="utf-8-sig")