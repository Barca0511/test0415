import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

SUMMER_MONTHS = [6,7,8]

# =========================
# Compound 新分箱（从负值开始）
# =========================

BIN_EDGES = [-np.inf,-1,0,1,2,3,4,np.inf]

BIN_LABELS = [
"bin_m1",
"bin_0",
"bin_1",
"bin_2",
"bin_3",
"bin_4",
"bin_5"
]

BASELINE_BIN = "bin_2"

BIN_MID_MAP = {
"bin_m1":-1.5,
"bin_0":-0.5,
"bin_1":0.5,
"bin_2":1.5,
"bin_3":2.5,
"bin_4":3.5,
"bin_5":4.5
}

BIN_TICK_LABELS = {
"bin_m1":"<=-1",
"bin_0":"(-1,0]",
"bin_1":"(0,1]",
"bin_2":"(1,2]",
"bin_3":"(2,3]",
"bin_4":"(3,4]",
"bin_5":">4"
}

def read_csv_auto(file_path):

    encodings=["utf-8-sig","utf-8","gbk","gb18030","latin1"]

    for enc in encodings:
        try:
            df=pd.read_csv(file_path,encoding=enc)
            print(f"{file_path} 读取成功，编码为: {enc}")
            return df
        except:
            pass

    raise ValueError("无法读取CSV")

def run_model(data,label):

    data=data.copy()

    data["temp_bin"]=pd.cut(
        data[TEMP_VAR],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        include_lowest=True
    )

    data=data.dropna(subset=["temp_bin"]).copy()

    bin_counts=data["temp_bin"].value_counts().reindex(BIN_LABELS,fill_value=0)

    print("\n各bin样本量：")
    print(bin_counts)

    dummies=pd.get_dummies(data["temp_bin"],prefix="tb",dtype=float)

    base=f"tb_{BASELINE_BIN}"

    if base in dummies.columns:
        dummies=dummies.drop(columns=[base])

    zero_cols=[c for c in dummies.columns if dummies[c].sum()==0]

    if zero_cols:
        print("删除空bin:",zero_cols)
        dummies=dummies.drop(columns=zero_cols)

    data=pd.concat([data,dummies],axis=1)

    rhs=list(dummies.columns)+CONTROL_VARS

    formula=f"{Y_VAR} ~ 0 + {' + '.join(rhs)} + EntityEffects + TimeEffects"

    panel=data.set_index([CITY_VAR,DATE_VAR]).sort_index()

    model=PanelOLS.from_formula(formula,data=panel)

    res=model.fit(cov_type="clustered",cluster_entity=True)

    rows=[]

    for b in BIN_LABELS:

        if b==BASELINE_BIN:

            coef=0
            low=0
            high=0

        else:

            name=f"tb_{b}"

            if name in res.params.index:

                coef=res.params[name]
                se=res.std_errors[name]

                low=coef-1.96*se
                high=coef+1.96*se

            else:

                coef=np.nan
                low=np.nan
                high=np.nan

        rows.append({

        "bin":b,
        "x":BIN_MID_MAP[b],
        "coef":coef*100,
        "low":low*100,
        "high":high*100,
        "count":int(bin_counts[b]),
        "label":BIN_TICK_LABELS[b]

        })

    coef_df=pd.DataFrame(rows)

    return res,coef_df,data

df=read_csv_auto(FILE_PATH)

df.columns=df.columns.str.strip()

df[DATE_VAR]=pd.to_datetime(df[DATE_VAR],errors="coerce")

needed=[CITY_VAR,DATE_VAR,Y_VAR,TEMP_VAR,COUNT_VAR,"day","night"]+CONTROL_VARS

missing=[c for c in needed if c not in df.columns]

if missing:
    raise ValueError(f"缺少列 {missing}")

for col in [Y_VAR,TEMP_VAR,COUNT_VAR,"day","night"]+CONTROL_VARS:
    df[col]=pd.to_numeric(df[col],errors="coerce")

df=df.dropna(subset=[CITY_VAR,DATE_VAR,Y_VAR,TEMP_VAR,"day","night"]+CONTROL_VARS)

df=df[df[DATE_VAR].dt.month.isin(SUMMER_MONTHS)]

city_counts=df.groupby(CITY_VAR)[COUNT_VAR].sum().sort_values(ascending=False)

top_cities=city_counts.head(130).index

df=df[df[CITY_VAR].isin(top_cities)]

# Compound 样本

df=df[(df["day"]==1)&(df["night"]==1)]

print("\nCompound 样本量:",len(df))

res,coef,used_data=run_model(df,"Compound")

print("\n===== 回归结果 =====")
print(res)

print("\n===== 绘图数据 =====")
print(coef)

plt.figure(figsize=(10,6))

x=coef["x"]
y=coef["coef"]

plt.plot(x,y,lw=2,marker="o")

plt.fill_between(x,coef["low"],coef["high"],alpha=0.2)

plt.axhline(0,ls="--",color="grey")

plt.title("Compound Heat")

plt.xlabel("Temperature anomaly (°C)")
plt.ylabel("Sentiment change (%)")

plt.xticks(coef["x"],coef["label"],rotation=30)

for _,row in coef.iterrows():

    plt.text(row["x"],row["coef"],f"n={row['count']}",ha="center")

plt.tight_layout()

plt.savefig("compound_regression_bins_negative_start.png",dpi=300)

plt.show()

print("\n最终bin样本量")
print(coef[["label","count"]])