import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

# =========================
# 基本参数
# =========================
FILE_PATH = "final_max.csv"

CITY_VAR = "city"
DATE_VAR = "date"

Y_VAR = "mean_large_model"
TEMP_VAR = "temp_max_anomaly"
COUNT_VAR = "count"

CONTROL_VARS = [
    "precip_mm_mean",
    "rh_mean",
    "sp_hpa_mean",
    "wind_speed_mean"
]

SUMMER_MONTHS = [6, 7, 8]

# =========================
# 更稳健的分箱方案
# baseline 设为 <=2
# =========================
BIN_EDGES = [-np.inf, 2, 2.5, 3, 3.5, 4, 5, np.inf]
BIN_LABELS = [
    "bin_le2",     # <= 2
    "bin_2_25",    # (2, 2.5]
    "bin_25_3",    # (2.5, 3]
    "bin_3_35",    # (3, 3.5]
    "bin_35_4",    # (3.5, 4]
    "bin_4_5",     # (4, 5]
    "bin_gt5"      # > 5
]

BASELINE_BIN = "bin_2_25"

BIN_MID_MAP = {
    "bin_le2": 1.0,
    "bin_2_25": 2.25,
    "bin_25_3": 2.75,
    "bin_3_35": 3.25,
    "bin_35_4": 3.75,
    "bin_4_5": 4.5,
    "bin_gt5": 5.5
}

BIN_TICK_LABELS = {
    "bin_le2": "<=2",
    "bin_2_25": "(2,2.5]",
    "bin_25_3": "(2.5,3]",
    "bin_3_35": "(3,3.5]",
    "bin_35_4": "(3.5,4]",
    "bin_4_5": "(4,5]",
    "bin_gt5": ">5"
}

# =========================
# 读文件
# =========================
def read_csv_auto(file_path):
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_error = None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {os.path.abspath(file_path)}")

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"{file_path} 读取成功，编码为: {enc}")
            return df
        except Exception as e:
            last_error = e

    raise last_error

# =========================
# 回归函数
# =========================
def run_model(data, label):
    data = data.copy()

    data["temp_bin"] = pd.cut(
        data[TEMP_VAR],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        include_lowest=True
    )

    data = data.dropna(subset=["temp_bin"]).copy()

    if len(data) == 0:
        raise ValueError(f"{label} 分组在分箱后没有可用样本")

    bin_counts = data["temp_bin"].value_counts().reindex(BIN_LABELS, fill_value=0)
    print(f"\n{label} 各bin样本量：")
    print(bin_counts)

    dummies = pd.get_dummies(data["temp_bin"], prefix="tb", dtype=float)

    base = f"tb_{BASELINE_BIN}"
    if base in dummies.columns:
        dummies = dummies.drop(columns=[base])

    zero_cols = [c for c in dummies.columns if dummies[c].sum() == 0]
    if zero_cols:
        print(f"\n{label} 删除全0 dummy 列：{zero_cols}")
        dummies = dummies.drop(columns=zero_cols)

    data = pd.concat([data, dummies], axis=1)

    rhs = list(dummies.columns) + CONTROL_VARS
    if len(rhs) == 0:
        raise ValueError(f"{label} 分组没有可用于回归的自变量")

    formula = f"{Y_VAR} ~ 0 + {' + '.join(rhs)} + EntityEffects + TimeEffects"

    panel = data.set_index([CITY_VAR, DATE_VAR]).sort_index()

    model = PanelOLS.from_formula(formula, data=panel)
    res = model.fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for b in BIN_LABELS:
        if b == BASELINE_BIN:
            coef = 0.0
            low = 0.0
            high = 0.0
        else:
            name = f"tb_{b}"
            if name in res.params.index:
                coef = res.params[name]
                se = res.std_errors[name]
                low = coef - 1.96 * se
                high = coef + 1.96 * se
            else:
                coef = np.nan
                low = np.nan
                high = np.nan

        rows.append({
            "bin": b,
            "x": BIN_MID_MAP[b],
            "coef": coef * 100,
            "low": low * 100,
            "high": high * 100,
            "count": int(bin_counts[b]),
            "label": BIN_TICK_LABELS[b]
        })

    coef_df = pd.DataFrame(rows)
    return res, coef_df, data

# =========================
# 主程序
# =========================
df = read_csv_auto(FILE_PATH)

df.columns = df.columns.str.strip()
df[DATE_VAR] = pd.to_datetime(df[DATE_VAR], errors="coerce")

needed_cols = [CITY_VAR, DATE_VAR, Y_VAR, TEMP_VAR, COUNT_VAR, "day", "night"] + CONTROL_VARS
missing_cols = [c for c in needed_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"缺少必要列: {missing_cols}")

for col in [Y_VAR, TEMP_VAR, COUNT_VAR, "day", "night"] + CONTROL_VARS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[CITY_VAR, DATE_VAR, Y_VAR, TEMP_VAR, "day", "night"] + CONTROL_VARS).copy()

df = df[df[DATE_VAR].dt.month.isin(SUMMER_MONTHS)].copy()

city_counts = df.groupby(CITY_VAR)[COUNT_VAR].sum().sort_values(ascending=False)
top_cities = city_counts.head(130).index
df = df[df[CITY_VAR].isin(top_cities)].copy()

df = df[(df["day"] == 1) & (df["night"] == 0)].copy()

print(f"\nDayOnly 样本量: {len(df)}")

res, coef, used_data = run_model(df, "DayOnly")

print("\n===== 回归结果 =====")
print(res)

print("\n===== 绘图用系数表 =====")
print(coef)

# =========================
# 绘图
# =========================
plt.figure(figsize=(10, 6))

x = coef["x"]
y = coef["coef"]

plt.plot(x, y, lw=2, marker="o")
plt.fill_between(x, coef["low"], coef["high"], alpha=0.2)

plt.axhline(0, ls="--", color="grey")

plt.title("DayOnly: Temperature Anomaly Bins")
plt.xlabel("Temperature anomaly (°C)")
plt.ylabel("Sentiment change (%)")

plt.xticks(coef["x"], coef["label"], rotation=30, ha="right")

for _, row in coef.iterrows():
    plt.text(row["x"], row["coef"], f"n={row['count']}", fontsize=9, ha="center", va="bottom")

plt.tight_layout()
plt.savefig("dayonly_regression_better_bins.png", dpi=300)
plt.show()

print("\n===== 最终各bin样本量汇总 =====")
print(coef[["label", "count"]])