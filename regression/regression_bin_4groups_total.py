import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from linearmodels.panel import PanelOLS

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

BIN_EDGES = [-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf]
BIN_LABELS = [
    "bin_m3", "bin_m2", "bin_m1", "bin_0",
    "bin_1", "bin_2", "bin_3", "bin_4"
]

BASELINE_BIN = "bin_0"

BIN_MID_MAP = {
    "bin_m3": -3.5,
    "bin_m2": -2.5,
    "bin_m1": -1.5,
    "bin_0": -0.5,
    "bin_1": 0.5,
    "bin_2": 1.5,
    "bin_3": 2.5,
    "bin_4": 4
}

def read_csv_auto(file_path):
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"{file_path} 读取成功，编码为: {enc}")
            return df
        except Exception as e:
            last_error = e
    raise last_error

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

# city_counts=df.groupby(CITY_VAR)[COUNT_VAR].sum().sort_values(ascending=False)
# top_cities=city_counts.head(130).index
#
# df=df[df[CITY_VAR].isin(top_cities)]

groups = {
    "DayOnly": (df["day"] == 1) & (df["night"] == 0),
    "NightOnly": (df["day"] == 0) & (df["night"] == 1),
    "Compound": (df["day"] == 1) & (df["night"] == 1),
    "Normal": (df["day"] == 0) & (df["night"] == 0)
}

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

    dummies = pd.get_dummies(data["temp_bin"], prefix="tb", dtype=float)

    base = f"tb_{BASELINE_BIN}"
    if base in dummies.columns:
        dummies = dummies.drop(columns=[base])

    zero_cols = [c for c in dummies.columns if dummies[c].sum() == 0]
    if zero_cols:
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
            coef = 0
            low = 0
            high = 0
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
            "high": high * 100
        })

    coef = pd.DataFrame(rows)

    return coef, data[TEMP_VAR]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

i = 0

for name, cond in groups.items():
    sub = df[cond].copy()
    ax = axs.flatten()[i]

    if len(sub) == 0:
        ax.set_title(name)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Temperature anomaly (°C)")
        ax.set_ylabel("Sentiment change (%)")
        i += 1
        continue

    try:
        coef, temp = run_model(sub, name)

        x = coef["x"]
        y = coef["coef"]

        ax.plot(x, y, color="#58c4dd", lw=2)
        ax.scatter(x, y, color="#58c4dd")

        ax.fill_between(x, coef["low"], coef["high"], alpha=0.2)

        ax.axhline(0, ls="--", color="grey")

        ax.set_title(name)

        ax.set_xlabel("Temperature anomaly (°C)")
        ax.set_ylabel("Sentiment change (%)")
    except Exception as e:
        ax.set_title(name)
        ax.text(0.5, 0.5, f"Model failed:\n{e}", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Temperature anomaly (°C)")
        ax.set_ylabel("Sentiment change (%)")

    i += 1

plt.tight_layout()
plt.savefig("anomaly_regression_top130.png", dpi=300)
plt.show()