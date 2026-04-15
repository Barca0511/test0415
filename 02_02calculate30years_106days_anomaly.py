# import pandas as pd
# import os
# from glob import glob
#
# input_folder = r"GEE_Tmax"
# file_climate = r"history_mean(Tmax_Tmean_Tmin)/ERA5Land_1981_2010_summer_tmax_mean30_by_diming_mean.csv"
# output_folder = r"2023/history_anomaly/Tmax"
#
# os.makedirs(output_folder, exist_ok=True)
#
# file_pattern = os.path.join(input_folder, "ERA5Land_*_daily_tmax_by_diming_mean.csv")
# file_list = sorted(glob(file_pattern))
#
# df_climate = pd.read_csv(file_climate)
#
# for file_path in file_list:
#     df = pd.read_csv(file_path)
#
#     df["date"] = pd.to_datetime(df["date"], errors="coerce")
#
#     df = df[
#         (
#             ((df["date"].dt.month == 5) & (df["date"].dt.day >= 25)) |
#             (df["date"].dt.month == 6) |
#             (df["date"].dt.month == 7) |
#             (df["date"].dt.month == 8) |
#             ((df["date"].dt.month == 9) & (df["date"].dt.day <= 7))
#         )
#     ].copy()
#
#     df = df.merge(df_climate[["region_name", "temp_max_mean30"]], on="region_name", how="left")
#
#     df["temp_max_anomaly"] = df["temp_max_c"] - df["temp_max_mean30"]
#
#     result = df[["region_name", "date", "temp_max_anomaly"]]
#
#     output_name = os.path.basename(file_path).replace(
#         "_daily_tmax_by_diming_mean.csv",
#         "_tmax_anomaly_0525_0907_by_diming.csv"
#     )
#     output_file = os.path.join(output_folder, output_name)
#
#     result.to_csv(output_file, index=False, encoding="utf-8-sig")







# import pandas as pd
# import os
# from glob import glob
#
# input_folder = r"GEE_Tmean"
# file_climate = r"history_mean(Tmax_Tmean_Tmin)/ERA5Land_1981_2010_summer_tmean_mean30_by_diming_mean.csv"
# output_folder = r"2023/history_anomaly/Tmean"
#
# os.makedirs(output_folder, exist_ok=True)
#
# file_pattern = os.path.join(input_folder, "ERA5Land_*_daily_tmean_by_diming_mean.csv")
# file_list = sorted(glob(file_pattern))
#
# df_climate = pd.read_csv(file_climate)
#
# for file_path in file_list:
#     df = pd.read_csv(file_path)
#
#     df["date"] = pd.to_datetime(df["date"], errors="coerce")
#
#     df = df[
#         (
#             ((df["date"].dt.month == 5) & (df["date"].dt.day >= 25)) |
#             (df["date"].dt.month == 6) |
#             (df["date"].dt.month == 7) |
#             (df["date"].dt.month == 8) |
#             ((df["date"].dt.month == 9) & (df["date"].dt.day <= 7))
#         )
#     ].copy()
#
#     df = df.merge(df_climate[["region_name", "temp_mean_mean30"]], on="region_name", how="left")
#
#     df["temp_mean_anomaly"] = df["temp_mean_c"] - df["temp_mean_mean30"]
#
#     result = df[["region_name", "date", "temp_mean_anomaly"]]
#
#     output_name = os.path.basename(file_path).replace(
#         "_daily_tmean_by_diming_mean.csv",
#         "_tmean_anomaly_0525_0907_by_diming.csv"
#     )
#     output_file = os.path.join(output_folder, output_name)
#
#     result.to_csv(output_file, index=False, encoding="utf-8-sig")







import pandas as pd
import os
from glob import glob

input_folder = r"GEE_Tmin"
file_climate = r"history_mean(Tmax_Tmean_Tmin)/ERA5Land_1981_2010_summer_tmin_mean30_by_diming_mean.csv"
output_folder = r"2023/history_anomaly/Tmin"

os.makedirs(output_folder, exist_ok=True)

file_pattern = os.path.join(input_folder, "ERA5Land_*_daily_tmin_by_diming_mean.csv")
file_list = sorted(glob(file_pattern))

df_climate = pd.read_csv(file_climate)

for file_path in file_list:
    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[
        (
            ((df["date"].dt.month == 5) & (df["date"].dt.day >= 25)) |
            (df["date"].dt.month == 6) |
            (df["date"].dt.month == 7) |
            (df["date"].dt.month == 8) |
            ((df["date"].dt.month == 9) & (df["date"].dt.day <= 7))
        )
    ].copy()

    df = df.merge(df_climate[["region_name", "temp_min_mean30"]], on="region_name", how="left")

    df["temp_min_anomaly"] = df["temp_min_c"] - df["temp_min_mean30"]

    result = df[["region_name", "date", "temp_min_anomaly"]]

    output_name = os.path.basename(file_path).replace(
        "_daily_tmin_by_diming_mean.csv",
        "_tmin_anomaly_0525_0907_by_diming.csv"
    )
    output_file = os.path.join(output_folder, output_name)

    result.to_csv(output_file, index=False, encoding="utf-8-sig")