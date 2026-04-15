# Tmax

import pandas as pd
import os

file_2023 = r"2023/ERA5Land_2023_daily_tmax_by_diming_mean.csv"
file_climate = r"history_mean(Tmax_Tmean_Tmin)/ERA5Land_1981_2010_summer_tmax_mean30_by_diming_mean.csv"
output_file = r"2023/ERA5Land_2023_tmax_anomaly_0525_0907_by_diming.csv"

df2023 = pd.read_csv(file_2023)
df_climate = pd.read_csv(file_climate)

df2023["date"] = pd.to_datetime(df2023["date"], errors="coerce")

df2023 = df2023[
    (
        ((df2023["date"].dt.month == 5) & (df2023["date"].dt.day >= 25)) |
        (df2023["date"].dt.month == 6) |
        (df2023["date"].dt.month == 7) |
        (df2023["date"].dt.month == 8) |
        ((df2023["date"].dt.month == 9) & (df2023["date"].dt.day <= 7))
    )
].copy()

df = df2023.merge(df_climate[["region_name", "temp_max_mean30"]], on="region_name", how="left")

df["temp_max_anomaly"] = df["temp_max_c"] - df["temp_max_mean30"]

result = df[["region_name", "date", "temp_max_anomaly"]]

result.to_csv(output_file, index=False, encoding="utf-8-sig")











#Tmean

# import pandas as pd
# import os
#
# file_2023 = r"2023/ERA5Land_2023_daily_tmean_by_diming_mean.csv"
# file_climate = r"history_mean(Tmax_Tmean_Tmin)/ERA5Land_1981_2010_summer_tmean_mean30_by_diming_mean.csv"
# output_file = r"2023/ERA5Land_2023_tmean_anomaly_0525_0907_by_diming.csv"
#
# df2023 = pd.read_csv(file_2023)
# df_climate = pd.read_csv(file_climate)
#
# df2023["date"] = pd.to_datetime(df2023["date"], errors="coerce")
#
# df2023 = df2023[
#     (
#         ((df2023["date"].dt.month == 5) & (df2023["date"].dt.day >= 25)) |
#         (df2023["date"].dt.month == 6) |
#         (df2023["date"].dt.month == 7) |
#         (df2023["date"].dt.month == 8) |
#         ((df2023["date"].dt.month == 9) & (df2023["date"].dt.day <= 7))
#     )
# ].copy()
#
# df = df2023.merge(df_climate[["region_name", "temp_mean_mean30"]], on="region_name", how="left")
#
# df["temp_mean_anomaly"] = df["temp_mean_c"] - df["temp_mean_mean30"]
#
# result = df[["region_name", "date", "temp_mean_anomaly"]]
#
# result.to_csv(output_file, index=False, encoding="utf-8-sig")









#Tmin
# import pandas as pd
# import os
#
# file_2023 = r"2023/ERA5Land_2023_daily_tmin_by_diming_mean.csv"
# file_climate = r"history_mean(Tmax_Tmean_Tmin)/ERA5Land_1981_2010_summer_tmin_mean30_by_diming_mean.csv"
# output_file = r"2023/ERA5Land_2023_tmin_anomaly_0525_0907_by_diming.csv"
#
# df2023 = pd.read_csv(file_2023)
# df_climate = pd.read_csv(file_climate)
#
# df2023["date"] = pd.to_datetime(df2023["date"], errors="coerce")
#
# df2023 = df2023[
#     (
#         ((df2023["date"].dt.month == 5) & (df2023["date"].dt.day >= 25)) |
#         (df2023["date"].dt.month == 6) |
#         (df2023["date"].dt.month == 7) |
#         (df2023["date"].dt.month == 8) |
#         ((df2023["date"].dt.month == 9) & (df2023["date"].dt.day <= 7))
#     )
# ].copy()
#
# df = df2023.merge(df_climate[["region_name", "temp_min_mean30"]], on="region_name", how="left")
#
# df["temp_min_anomaly"] = df["temp_min_c"] - df["temp_min_mean30"]
#
# result = df[["region_name", "date", "temp_min_anomaly"]]
#
# result.to_csv(output_file, index=False, encoding="utf-8-sig")


