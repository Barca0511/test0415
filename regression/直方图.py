import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = "final_max.csv"

CITY_VAR = "city"
DATE_VAR = "date"
Y_VAR = "mean_large_model"

SUMMER_MONTHS = [6, 7, 8]

def read_csv_auto(file_path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"{file_path} 读取成功，编码为: {enc}")
            return df
        except Exception:
            pass
    raise ValueError("无法读取文件")

df = read_csv_auto(FILE_PATH)

df.columns = df.columns.str.strip()
df[DATE_VAR] = pd.to_datetime(df[DATE_VAR], errors="coerce")

df = df[df[DATE_VAR].dt.month.isin(SUMMER_MONTHS)].copy()
df = df.dropna(subset=[Y_VAR, "day", "night"]).copy()

df["day"] = pd.to_numeric(df["day"], errors="coerce")
df["night"] = pd.to_numeric(df["night"], errors="coerce")
df[Y_VAR] = pd.to_numeric(df[Y_VAR], errors="coerce")

df = df.dropna(subset=[Y_VAR, "day", "night"]).copy()

groups = {
    "DayOnly": df[(df["day"] == 1) & (df["night"] == 0)].copy(),
    "NightOnly": df[(df["day"] == 0) & (df["night"] == 1)].copy(),
    "Compound": df[(df["day"] == 1) & (df["night"] == 1)].copy(),
    "Normal": df[(df["day"] == 0) & (df["night"] == 0)].copy()
}

print("\n各组样本量：")
for name, sub in groups.items():
    print(f"{name}: {len(sub)}")

# ===== 输出统计量 =====

print("\n情绪统计量（mean / std / variance）:")

for name, sub in groups.items():

    values = sub[Y_VAR]

    mean_val = values.mean()
    std_val = values.std()
    var_val = values.var()

    print(f"\n{name}")
    print(f"mean     = {mean_val:.6f}")
    print(f"std      = {std_val:.6f}")
    print(f"variance = {var_val:.6f}")

# ===== 统一x轴范围 =====

all_values = pd.concat([sub[Y_VAR] for sub in groups.values() if len(sub) > 0], ignore_index=True)

x_min = all_values.min()
x_max = all_values.max()

bins = 50

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axes = axs.flatten()

for i, (name, sub) in enumerate(groups.items()):
    ax = axes[i]

    if len(sub) == 0:
        ax.set_title(name)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Sentiment score")
        ax.set_ylabel("Frequency")
        continue

    ax.hist(
        sub[Y_VAR],
        bins=bins,
        range=(x_min, x_max),
        edgecolor="black",
        alpha=0.75
    )

    ax.set_title(f"{name} (n={len(sub)})")
    ax.set_xlabel("Sentiment score")
    ax.set_ylabel("Frequency")
    ax.set_xlim(x_min, x_max)

plt.tight_layout()
plt.savefig("sentiment_hist_4panels.png", dpi=300)
plt.show()