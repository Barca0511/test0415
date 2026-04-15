#此代码用于测试如何分bin 每个bin内有多少样本
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

FILE_PATH = "final_max.csv"

CITY_VAR = "city"
DATE_VAR = "date"
TEMP_VAR = "temp_max_anomaly"

SUMMER_MONTHS = [6, 7, 8]

MIN_BIN_COUNT = 200

INITIAL_EDGES = [-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf]

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

def make_bin_labels(edges):
    labels = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]

        if np.isneginf(left):
            labels.append(f"bin_le_{right:g}")
        elif np.isposinf(right):
            labels.append(f"bin_gt_{left:g}")
        else:
            labels.append(f"bin_{left:g}_{right:g}".replace("-", "m").replace(".", "p"))
    return labels

def make_bin_mid_map(edges, labels):
    mid_map = {}
    finite_edges = [x for x in edges if np.isfinite(x)]
    if len(finite_edges) >= 2:
        typical_width = np.median(np.diff(finite_edges))
    else:
        typical_width = 1.0

    for i, lab in enumerate(labels):
        left = edges[i]
        right = edges[i + 1]

        if np.isneginf(left) and np.isfinite(right):
            mid = right - typical_width / 2
        elif np.isfinite(left) and np.isposinf(right):
            mid = left + typical_width / 2
        else:
            mid = (left + right) / 2

        mid_map[lab] = mid
    return mid_map

def count_bins_for_group(series, edges):
    labels = make_bin_labels(edges)
    binned = pd.cut(series, bins=edges, labels=labels, include_lowest=True, right=True)
    counts = binned.value_counts().reindex(labels, fill_value=0)
    return counts

def choose_merge_index(group_counts_dict, labels):
    score = pd.DataFrame(group_counts_dict).copy()
    score["min_across_groups"] = score.min(axis=1)

    min_idx = score["min_across_groups"].idxmin()
    pos = labels.index(min_idx)

    if pos == 0:
        return 0
    if pos == len(labels) - 1:
        return len(labels) - 2

    left_label = labels[pos - 1]
    right_label = labels[pos + 1]

    left_total = score.loc[min_idx, "min_across_groups"] + score.loc[left_label, "min_across_groups"]
    right_total = score.loc[min_idx, "min_across_groups"] + score.loc[right_label, "min_across_groups"]

    if left_total >= right_total:
        return pos - 1
    else:
        return pos

def auto_merge_edges(df_groups, temp_var, initial_edges, min_bin_count):
    edges = initial_edges.copy()

    while True:
        labels = make_bin_labels(edges)
        group_counts_dict = {}

        for group_name, subdf in df_groups.items():
            counts = count_bins_for_group(subdf[temp_var], edges)
            group_counts_dict[group_name] = counts

        count_df = pd.DataFrame(group_counts_dict).reindex(labels)

        print("\n当前 bin 统计：")
        print(count_df)

        enough = (count_df >= min_bin_count).all().all()

        if enough:
            return edges, count_df

        if len(edges) <= 3:
            print("\n已经无法继续合并。")
            return edges, count_df

        merge_idx = choose_merge_index(group_counts_dict, labels)
        removed_edge = edges[merge_idx + 1]
        print(f"\n样本过少，自动合并，删除切点: {removed_edge}")
        edges.pop(merge_idx + 1)

def print_bin_code(edges):
    labels = make_bin_labels(edges)
    mid_map = make_bin_mid_map(edges, labels)

    print("\n推荐 BIN_EDGES =")
    print(edges)

    print("\n推荐 BIN_LABELS =")
    print(labels)

    print("\n推荐 BIN_MID_MAP =")
    print("{")
    for k, v in mid_map.items():
        print(f'    "{k}": {v},')
    print("}")

df = read_csv_auto(FILE_PATH)
df.columns = df.columns.str.strip()
df[DATE_VAR] = pd.to_datetime(df[DATE_VAR], errors="coerce")

needed_cols = [CITY_VAR, DATE_VAR, TEMP_VAR, "day", "night"]
missing_cols = [c for c in needed_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"缺少必要列: {missing_cols}")

for col in [TEMP_VAR, "day", "night"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[CITY_VAR, DATE_VAR, TEMP_VAR, "day", "night"]).copy()
df = df[df[DATE_VAR].dt.month.isin(SUMMER_MONTHS)].copy()

groups = {
    "DayOnly": df[(df["day"] == 1) & (df["night"] == 0)].copy(),
    "NightOnly": df[(df["day"] == 0) & (df["night"] == 1)].copy(),
    "Compound": df[(df["day"] == 1) & (df["night"] == 1)].copy(),
    "Normal": df[(df["day"] == 0) & (df["night"] == 0)].copy()
}

print("\n各组样本量：")
for k, v in groups.items():
    print(f"{k}: {len(v)}")

final_edges, final_count_df = auto_merge_edges(
    df_groups=groups,
    temp_var=TEMP_VAR,
    initial_edges=INITIAL_EDGES,
    min_bin_count=MIN_BIN_COUNT
)

print("\n最终推荐 bins：")
print_bin_code(final_edges)

print("\n最终各组 bin 样本量：")
print(final_count_df)