import pandas as pd
import os
from glob import glob

# =========================
# 1. 设置文件夹路径
# =========================
# 改成你的30个CSV所在文件夹
input_folder = r"GEE_Tmax"

# 输出文件
output_file = os.path.join(input_folder, "ERA5Land_1981_2010_summer_tmax_mean30_by_diming.csv")

# 匹配文件
file_pattern = os.path.join(input_folder, "ERA5Land_*_daily_tmax_by_diming_mean.csv")
file_list = sorted(glob(file_pattern))

if len(file_list) == 0:
    raise FileNotFoundError("没有找到符合命名规则的CSV文件，请检查路径和文件名。")

print(f"共找到 {len(file_list)} 个文件。")

# =========================
# 2. 逐年读取并计算“每个地区每年夏季均值”
# =========================
yearly_summer_means = []

for file in file_list:
    print(f"正在处理: {os.path.basename(file)}")

    df = pd.read_csv(file)

    # 检查字段
    required_cols = {"region_name", "date", "temp_max_c"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{os.path.basename(file)} 缺少必要列，必须包含: {required_cols}")

    # 日期转为 datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 删除无效日期或无效温度
    df = df.dropna(subset=["date", "temp_max_c"])

    # 取夏季：6月1日 - 8月31日
    summer_df = df[
        (
            ((df["date"].dt.month == 6) & (df["date"].dt.day >= 1)) |
            (df["date"].dt.month == 7) |
            ((df["date"].dt.month == 8) & (df["date"].dt.day <= 31))
        )
    ].copy()

    # 提取年份
    summer_df["year"] = summer_df["date"].dt.year

    # 每个地区、每一年，求整个夏季平均值
    yearly_mean = (
        summer_df
        .groupby(["region_name", "year"], as_index=False)["temp_max_c"]
        .mean()
        .rename(columns={"temp_max_c": "summer_mean"})
    )

    yearly_summer_means.append(yearly_mean)

# 合并30年结果
all_yearly_means = pd.concat(yearly_summer_means, ignore_index=True)

# =========================
# 3. 对30年的夏季均值再求平均
# =========================
result = (
    all_yearly_means
    .groupby("region_name", as_index=False)["summer_mean"]
    .mean()
    .rename(columns={"summer_mean": "temp_max_mean30"})
)

# 加一个季节标签列
result["date"] = "06-01_to_08-31"

# 调整列顺序
result = result[["region_name", "date", "temp_max_mean30"]]

# 保存
result.to_csv(output_file, index=False, encoding="utf-8-sig")

print("处理完成，结果已保存到：")
print(output_file)
print(result.head())