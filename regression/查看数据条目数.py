import pandas as pd

FILE_PATH = "final_max.csv"

CITY_VAR = "city"
DATE_VAR = "date"

df = pd.read_csv(FILE_PATH, encoding="gb18030")

df.columns = df.columns.str.strip()

df[DATE_VAR] = pd.to_datetime(df[DATE_VAR], errors="coerce")

# 只保留夏季
df = df[df[DATE_VAR].dt.month.isin([6,7,8])].copy()

total = len(df)

day_only = df[(df["day"] == 1) & (df["night"] == 0)]
night_only = df[(df["day"] == 0) & (df["night"] == 1)]
compound = df[(df["day"] == 1) & (df["night"] == 1)]
normal = df[(df["day"] == 0) & (df["night"] == 0)]

print("\n===== 样本数量统计 =====\n")

print(f"夏季总样本数: {total}\n")

print(f"单一日间高温 DayOnly: {len(day_only)}  ({len(day_only)/total:.2%})")
print(f"单一夜间高温 NightOnly: {len(night_only)}  ({len(night_only)/total:.2%})")
print(f"复合高温 Compound: {len(compound)}  ({len(compound)/total:.2%})")
print(f"正常 Normal: {len(normal)}  ({len(normal)/total:.2%})")

print("\n===== sanity check =====")

check = len(day_only) + len(night_only) + len(compound) + len(normal)

print(f"四类样本合计: {check}")
print(f"是否等于总样本: {check == total}")

print("\n===== 每个城市的复合高温数量（前20）=====")

compound_city = compound.groupby(CITY_VAR).size().sort_values(ascending=False)

print(compound_city.head(20))