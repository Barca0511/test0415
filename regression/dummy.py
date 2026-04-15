import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

FILE_PATH = "final_max.csv"

CITY_VAR = "city"
DATE_VAR = "date"
Y_VAR = "mean_large_model"
TEMP_VAR = "temp_max_anomaly"

COUNT_VAR = "count"

DAY_VAR = "day"
NIGHT_VAR = "night"

CONTROL_VARS = ["precip_mm_mean", "rh_mean", "sp_hpa_mean", "wind_speed_mean"]

SUMMER_ONLY = True
SUMMER_MONTHS = [6, 7, 8]

OUTPUT_DIR = "dummy_fe_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULT_XLSX = os.path.join(OUTPUT_DIR, "dummy_model_results.xlsx")

BIN_EDGES = [-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf]
BIN_LABELS = ["bin_m3", "bin_m2", "bin_m1", "bin_0", "bin_1", "bin_2", "bin_3", "bin_4"]
BASELINE_BIN = "tb_bin_0"

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

needed_cols = [CITY_VAR, DATE_VAR, Y_VAR, TEMP_VAR, COUNT_VAR, DAY_VAR, NIGHT_VAR] + CONTROL_VARS
missing_cols = [c for c in needed_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"缺少必要列: {missing_cols}")

for col in [Y_VAR, TEMP_VAR, COUNT_VAR, DAY_VAR, NIGHT_VAR] + CONTROL_VARS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[CITY_VAR, DATE_VAR, Y_VAR, TEMP_VAR, DAY_VAR, NIGHT_VAR] + CONTROL_VARS).copy()

if SUMMER_ONLY:
    df = df[df[DATE_VAR].dt.month.isin(SUMMER_MONTHS)].copy()
    print(f"Only summer sample kept: months={SUMMER_MONTHS}, remaining rows={len(df)}")

# TOP_N = 130
# city_counts = df.groupby(CITY_VAR)[COUNT_VAR].sum().sort_values(ascending=False)
# top_cities = city_counts.head(TOP_N).index
# df = df[df[CITY_VAR].isin(top_cities)].copy()

df["day_heat"] = ((df[DAY_VAR] == 1) & (df[NIGHT_VAR] == 0)).astype(int)
df["night_heat"] = ((df[DAY_VAR] == 0) & (df[NIGHT_VAR] == 1)).astype(int)
df["compound_heat"] = ((df[DAY_VAR] == 1) & (df[NIGHT_VAR] == 1)).astype(int)
df["normal_heat"] = ((df[DAY_VAR] == 0) & (df[NIGHT_VAR] == 0)).astype(int)

print("\n各类样本量：")
print("DayOnly:", int(df["day_heat"].sum()))
print("NightOnly:", int(df["night_heat"].sum()))
print("Compound:", int(df["compound_heat"].sum()))
print("Normal:", int(df["normal_heat"].sum()))

df["temp_bin"] = pd.cut(df[TEMP_VAR], bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True)
temp_dummies = pd.get_dummies(df["temp_bin"], prefix="tb", dtype=float)

if BASELINE_BIN in temp_dummies.columns:
    temp_dummies = temp_dummies.drop(columns=[BASELINE_BIN])

zero_cols = [c for c in temp_dummies.columns if temp_dummies[c].sum() == 0]
if zero_cols:
    temp_dummies = temp_dummies.drop(columns=zero_cols)

df = pd.concat([df, temp_dummies], axis=1)

reg_cols = [Y_VAR, CITY_VAR, DATE_VAR] + list(temp_dummies.columns) + ["day_heat", "night_heat", "compound_heat"] + CONTROL_VARS
df = df.dropna(subset=reg_cols).copy()
df = df.set_index([CITY_VAR, DATE_VAR]).sort_index()

formula = (
    f"{Y_VAR} ~ 1 + "
    + " + ".join(list(temp_dummies.columns) + ["day_heat", "night_heat", "compound_heat"] + CONTROL_VARS)
    + " + EntityEffects + TimeEffects"
)

model = PanelOLS.from_formula(formula, data=df)
res = model.fit(cov_type="clustered", cluster_entity=True)

print("\n回归结果：")
print(res)

event_vars = ["day_heat", "night_heat", "compound_heat"]
event_coef_df = pd.DataFrame({
    "variable": event_vars,
    "coef": res.params[event_vars],
    "std_err": res.std_errors[event_vars],
    "pvalue": res.pvalues[event_vars],
    "ci_low": res.conf_int().loc[event_vars, "lower"],
    "ci_high": res.conf_int().loc[event_vars, "upper"]
}).reset_index(drop=True)

def plot_event_forest_plot(event_df, output_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    event_df = event_df.copy()
    event_df["coef_pct"] = event_df["coef"] * 100
    event_df["ci_low_pct"] = event_df["ci_low"] * 100
    event_df["ci_high_pct"] = event_df["ci_high"] * 100

    label_map = {
        "day_heat": "Day-only Heat",
        "night_heat": "Night-only Heat",
        "compound_heat": "Compound Heat"
    }
    event_df["label"] = event_df["variable"].map(label_map)

    colors = {
        "day_heat": "#F4C542",
        "night_heat": "#F28E2B",
        "compound_heat": "#D62728"
    }

    y_pos = np.arange(len(event_df))

    for i, row in event_df.iterrows():
        var = row["variable"]
        ax.errorbar(
            row["coef_pct"],
            y_pos[i],
            xerr=[[row["coef_pct"] - row["ci_low_pct"]], [row["ci_high_pct"] - row["coef_pct"]]],
            fmt="o",
            color=colors[var],
            markersize=10,
            capsize=5,
            elinewidth=2.5
        )
        ax.text(
            row["ci_high_pct"] + 0.03,
            y_pos[i],
            f"{row['coef_pct']:.3f}%",
            va="center",
            fontsize=10,
            color=colors[var]
        )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(event_df["label"], fontsize=12)
    ax.set_xlabel("Estimated Effect on Sentiment (%)", fontsize=12)
    ax.set_title("Impact of Heat Event Types on Public Sentiment", fontsize=14, pad=14)
    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Forest plot saved to: {output_path}")
    plt.show()

EVENT_FIG_PATH = os.path.join(OUTPUT_DIR, "event_forest_plot_en.png")
plot_event_forest_plot(event_coef_df, EVENT_FIG_PATH)

with pd.ExcelWriter(RESULT_XLSX) as writer:
    event_coef_df.to_excel(writer, sheet_name="event_results", index=False)
    full_res = pd.concat([res.params, res.std_errors, res.pvalues], axis=1)
    full_res.columns = ["coef", "std_err", "pvalue"]
    full_res.to_excel(writer, sheet_name="full_model_params")

print("All tasks completed successfully!")