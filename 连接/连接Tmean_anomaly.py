import re
import pandas as pd

file1 = r"final_max.csv"
file2 = r"ERA5Land_2023_tmean_anomaly_0525_0907_by_diming.csv" #这一列要动
output_file = r"final_max.csv"

encodings = ["gbk", "utf-8", "utf-8-sig", "gb18030", "latin1"]

def read_csv_auto(file_path):
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"{file_path} 读取成功，编码为: {enc}")
            return df
        except Exception as e:
            last_error = e
    raise last_error

df1 = read_csv_auto(file1)
df2 = read_csv_auto(file2)

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

df1["date"] = pd.to_datetime(df1["date"], errors="coerce")
df2["date"] = pd.to_datetime(df2["date"], errors="coerce")

SUFFIXES = [
    "特别行政区",
    "藏族羌族自治州", "哈尼族彝族自治州", "土家族苗族自治州", "傣族景颇族自治州",
    "蒙古族藏族自治州", "壮族苗族自治州", "苗族侗族自治州", "布依族苗族自治州",
    "柯尔克孜自治州", "朝鲜族自治州", "哈萨克自治州", "回族自治州",
    "藏族自治州", "彝族自治州", "傈僳族自治州", "傣族自治州", "蒙古自治州",
    "黎族苗族自治县", "黎族自治县",
    "自治州", "自治县", "自治旗",
    "地区", "盟", "林区",
    "市", "县", "区"
]

SUFFIXES = sorted(set(SUFFIXES), key=len, reverse=True)

def norm_region_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[（(].*?[）)]", "", s)

    changed = True
    while changed:
        changed = False
        for suf in SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                s = s[:-len(suf)]
                changed = True
                break
    return s

df1["_city_norm"] = df1["city"].apply(norm_region_name)
df2["_region_norm"] = df2["region_name"].apply(norm_region_name)

merge_cols = [
    "temp_mean_anomaly",
]

missing_cols = [c for c in merge_cols if c not in df2.columns]
if missing_cols:
    raise ValueError(f"ERA5Land_2023_tmean_anomaly_0525_0907_by_diming.csv 缺少这些字段: {missing_cols}")

df2_key = (
    df2.groupby(["_region_norm", "date"], as_index=False)[merge_cols]
       .mean()
)

result = df1.merge(
    df2_key,
    how="left",
    left_on=["_city_norm", "date"],
    right_on=["_region_norm", "date"]
)

result = result.drop(columns=[c for c in ["_city_norm", "_region_norm"] if c in result.columns])

result.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"结果已保存到: {output_file}")

for col in merge_cols:
    miss = result[col].isna().sum()
    print(f"{col} 未匹配行数: {miss}")