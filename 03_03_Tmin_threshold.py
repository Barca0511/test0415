import pandas as pd
import os
from glob import glob

input_folder = r"2023/history_anomaly/Tmin"
output_folder = r"2023/threshold"
output_file = os.path.join(output_folder, "ERA5Land_1981_2010_tmin_anomaly_threshold_p90_0601_0831_by_diming.csv")

os.makedirs(output_folder, exist_ok=True)

file_pattern = os.path.join(input_folder, "ERA5Land_*_tmin_anomaly_0525_0907_by_diming.csv")
file_list = sorted(glob(file_pattern))

all_data = []

for file_path in file_list:
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["region_name", "date", "temp_min_anomaly"]).copy()
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    all_data.append(df[["region_name", "date", "month_day", "temp_min_anomaly"]])

all_df = pd.concat(all_data, ignore_index=True)

target_dates = pd.date_range("2001-06-01", "2001-08-31", freq="D")
window_map = {}

for target_date in target_dates:
    window_days = pd.date_range(target_date - pd.Timedelta(days=7), target_date + pd.Timedelta(days=7), freq="D")
    window_map[target_date.strftime("%m-%d")] = set(window_days.strftime("%m-%d"))

results = []

for target_md, window_mds in window_map.items():
    window_df = all_df[all_df["month_day"].isin(window_mds)].copy()
    threshold_df = (
        window_df.groupby("region_name")["temp_min_anomaly"]
        .quantile(0.9)
        .reset_index()
        .rename(columns={"temp_min_anomaly": "temp_min_anomaly_threshold"})
    )
    threshold_df["date"] = target_md
    results.append(threshold_df)

result = pd.concat(results, ignore_index=True)
result = result[["region_name", "date", "temp_min_anomaly_threshold"]]

result.to_csv(output_file, index=False, encoding="utf-8-sig")
print(output_file)
print(result.head())