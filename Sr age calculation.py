import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# =========================
# 1. 读入 Excel
# =========================
in_path  = r"E:\data for python\0-32&U1516-1.xlsx"
out_path = r"E:\data for python\U1516_Sr_inverted_age_with_uncertainty.xlsx"

df = pd.read_excel(in_path)

print("Excel columns:", df.columns.tolist())

# =========================
# 2. 选择需要的列
# =========================
# Excel建议排列：
# 第1列：McArthur Age (Ma)
# 第2列：McArthur 87Sr/86Sr
# 第3列：McArthur 2SE ×10^6
# 第4列：U1516 Depth
# 第5列：U1516 87Sr/86Sr
# 第6列：U1516 2sigma

AGE_REF_COL   = df.columns[0]
SR_REF_COL    = df.columns[1]
REF_2SE_COL   = df.columns[2]

DEPTH_U_COL   = df.columns[3]
SR_U_COL      = df.columns[4]
SAMPLE_2S_COL = df.columns[5]

# ---------- McArthur reference curve ----------
age_ref = df[AGE_REF_COL].to_numpy()
sr_ref  = df[SR_REF_COL].to_numpy()
ref_2se = df[REF_2SE_COL].to_numpy()

mask_ref = (
    np.isfinite(age_ref) &
    np.isfinite(sr_ref) &
    np.isfinite(ref_2se)
)

age_ref = age_ref[mask_ref]
sr_ref  = sr_ref[mask_ref]
ref_2se = ref_2se[mask_ref]

# McArthur Table 5 reports "2 s.e. × 10^6"
# e.g. 1 means 0.000001
ref_2se = ref_2se * 1e-6

# ---------- U1516 samples ----------
depth_u1516 = df[DEPTH_U_COL].to_numpy()
sr_u1516    = df[SR_U_COL].to_numpy()
sample_2s   = df[SAMPLE_2S_COL].to_numpy()

mask_u = (
    np.isfinite(depth_u1516) &
    np.isfinite(sr_u1516) &
    np.isfinite(sample_2s)
)

depth_u1516 = depth_u1516[mask_u]
sr_u1516    = sr_u1516[mask_u]
sample_2s   = sample_2s[mask_u]

print("参考曲线点数:", len(age_ref))
print("U1516 样品数:", len(sr_u1516))

# =========================
# 3. 构建 McArthur 曲线
# =========================

# mean 87Sr/86Sr curve
sr_of_age = interp1d(
    age_ref,
    sr_ref,
    kind='cubic',
    fill_value="extrapolate"
)

# age-dependent 2SE
# 对 uncertainty 用 linear interpolation 更稳妥，
# 避免 cubic interpolation 产生不合理的负值/过冲
se_of_age = interp1d(
    age_ref,
    ref_2se,
    kind='linear',
    fill_value="extrapolate"
)

# 致密年龄网格
age_grid = np.linspace(
    age_ref.min(),
    age_ref.max(),
    20000
)

sr_grid = sr_of_age(age_grid)
se_grid = se_of_age(age_grid)

# McArthur reference uncertainty envelope
sr_ref_lower = sr_grid - se_grid
sr_ref_upper = sr_grid + se_grid

# =========================
# 4. 中心 Sr 年龄反演
# =========================

def invert_sr_to_age(sr_value, sr_grid, age_grid):
    """
    根据 McArthur mean curve，
    找到与样品 Sr ratio 最接近的年龄。
    """
    idx = np.argmin(np.abs(sr_grid - sr_value))
    return age_grid[idx]


ages_sr = np.array([
    invert_sr_to_age(s, sr_grid, age_grid)
    for s in sr_u1516
])

# =========================
# 5. 计算年龄不确定度
# =========================

def get_age_range(
    sr_value,
    sample_error,
    age_grid,
    sr_ref_lower,
    sr_ref_upper,
    central_age
):
    """
    样品区间：
        sr_value ± sample 2sigma

    参考曲线区间：
        McArthur mean Sr ± age-dependent 2SE

    找出两个 Sr 区间发生重叠的所有年龄，
    并返回包含 central_age 的连续重叠区间。
    """

    sample_lower = sr_value - sample_error
    sample_upper = sr_value + sample_error

    # 两个区间有重叠：
    overlap = (
        (sample_lower <= sr_ref_upper) &
        (sample_upper >= sr_ref_lower)
    )

    if not np.any(overlap):
        return np.nan, np.nan

    # -------------------------------------------------
    # 找出所有连续的 overlap 区间
    # 这是为了防止 Sr-age curve 非单调时，
    # 把两个彼此分离的年龄区间错误合并起来
    # -------------------------------------------------
    indices = np.where(overlap)[0]

    split_points = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, split_points)

    # 找到 central_age 所在/最接近的连续区间
    central_idx = np.argmin(np.abs(age_grid - central_age))

    selected_group = min(
        groups,
        key=lambda g: np.min(np.abs(g - central_idx))
    )

    age_lower = age_grid[selected_group].min()
    age_upper = age_grid[selected_group].max()

    return age_lower, age_upper


age_lower = []
age_upper = []

for sr, err, age_central in zip(
    sr_u1516,
    sample_2s,
    ages_sr
):
    lo, hi = get_age_range(
        sr,
        err,
        age_grid,
        sr_ref_lower,
        sr_ref_upper,
        age_central
    )

    age_lower.append(lo)
    age_upper.append(hi)

age_lower = np.array(age_lower)
age_upper = np.array(age_upper)

# =========================
# 6. 计算不对称年龄误差
# =========================

minus_error = ages_sr - age_lower
plus_error  = age_upper - ages_sr

# =========================
# 7. 输出结果
# =========================

df_out = pd.DataFrame({
    "Depth_m": depth_u1516,
    "Sr_87Sr_86Sr": sr_u1516,
    "Sample_2sigma": sample_2s,
    "Sr_age_Ma": ages_sr,
    "Age_lower_Ma": age_lower,
    "Age_upper_Ma": age_upper,
    "Age_minus_error_Ma": minus_error,
    "Age_plus_error_Ma": plus_error
})

# 统一保留合适位数
df_out["Sr_age_Ma"] = df_out["Sr_age_Ma"].round(3)
df_out["Age_lower_Ma"] = df_out["Age_lower_Ma"].round(3)
df_out["Age_upper_Ma"] = df_out["Age_upper_Ma"].round(3)
df_out["Age_minus_error_Ma"] = df_out["Age_minus_error_Ma"].round(3)
df_out["Age_plus_error_Ma"] = df_out["Age_plus_error_Ma"].round(3)

print("\n===== U1516 Sr ages and uncertainties =====")
print(df_out.head(10))

df_out.to_excel(out_path, index=False)

print(f"\n结果已保存到：{out_path}")

# =========================
# 8. 画图检查
# =========================

plt.figure()

# McArthur mean curve
plt.plot(
    age_grid,
    sr_grid,
    label="McArthur et al. (2025)"
)

# McArthur 2SE envelope
plt.fill_between(
    age_grid,
    sr_ref_lower,
    sr_ref_upper,
    alpha=0.2,
    label="Reference curve 2SE"
)

# U1516 samples
plt.errorbar(
    ages_sr,
    sr_u1516,
    yerr=sample_2s,
    fmt='o',
    markersize=3,
    capsize=2,
    label="U1516 samples (2σ)"
)

plt.gca().invert_xaxis()

plt.xlabel("Age (Ma)")
plt.ylabel("87Sr/86Sr")
plt.title("Sr-isotope ages and uncertainty")
plt.legend()
plt.tight_layout()
plt.show()
