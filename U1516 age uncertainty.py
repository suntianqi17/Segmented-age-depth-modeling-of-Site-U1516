import numpy as np
import matplotlib.pyplot as plt

# =============================
# 0. 一些基础参数（你可以改）
# =============================
n_real = 2000        # Monte Carlo 实现实验次数
np.random.seed(42)   # 固定随机种子，保证结果可重复

# ============================================================
# 1. 输入年龄控制点：深度、年龄，以及事件类型（mag/bio/sr）
# ============================================================
# 深度：使用你最终的 Adapt depth (mbsf)，从浅到深排序
z_ties = np.array([
    123.1,   # C3n.2r
    220.7,   # C4Ar.3r
    224.0,   # D. neohamatus
    245.3,   # G. nepenthes
    247.2,   # R. pseudoumbilicus  (H1 top)
    253.6,   # 87Sr/86Sr tie point (H1 base)
    267.0,   # G. dehiscens        (H2 top)
    267.0,   # S. delphix          (H2 base)
    279.0,   # Ch. altus           (H3 top)
    279.0,   # P. opima            (H3 base)
    340.0    # D. saipanensis
])

# 年龄 (Ma)
t_ties = np.array([
    4.8,   # C3n.2r
    9.8,   # C4Ar.3r
    10.5,  # D. neohamatus
    11.7,  # G. nepenthes
    12.8,  # R. pseudoumbilicus
    16.4,  # Sr age
    22.5,  # G. dehiscens
    23.1,  # S. delphix
    25.4,  # Ch. altus
    26.9,  # P. opima
    34.4   # D. saipanensis
])

# 事件类型：只为了给不同类型赋不同 σ
# 'mag' = 磁性界线, 'bio' = 生物事件, 'sr' = Sr 同位素年龄
tie_types = np.array([
    'mag',  # 4.8
    'mag',  # 9.8
    'bio',  # 10.5
    'bio',  # 11.7
    'bio',  # 12.8
    'sr',   # 16.4
    'bio',  # 22.5
    'bio',  # 23.1
    'bio',  # 25.4
    'bio',  # 26.9
    'bio'   # 34.4
])

# =====================================================================
# 2. 为每个 tie point 计算 1σ 年龄误差：
#    GTS2020 σ + 更精细的 Sr σ（由测量误差 + 全球曲线斜率得到）
# =====================================================================

# 2.1 先设置一个“基础 σ”：可根据 GTS2020 调整
sigma_mag_default = 0.05   # 磁性界线 1σ (Myr)，示例值
sigma_bio_default = 0.20   # 生物事件 1σ (Myr)，示例值

# 2.2 Sr 年龄 σ：用 Sr 测量误差 + 曲线斜率计算
# ------------------------------------------------
# 假设你有：
#   measured 87Sr/86Sr = 0.708698
#   measurement 2σ (ratio) = 0.000020   （举例：你实验室 SRM987 2σ ~ 1e-5 量级）
#   所以 1σ = 0.000020 / 2
#   在 16–17 Ma 附近 McArthur 曲线局部斜率 d(87Sr/86Sr) / d(age)（每 Myr）
#   例如 slope ≈ 4e-5 per Myr（这里只是示意，你应当用真正的斜率）
#
# 年龄误差换算公式：
#   sigma_age = sigma_ratio / |slope_dSr_dt|
#

sigma_ratio_2s = 0.000008    # 你根据真实测试填写
sigma_ratio_1s = sigma_ratio_2s / 2.0

slope_dSr_dt = -7.25e-5         # d(87Sr/86Sr)/dMyr，需根据 McArthur 曲线局部拟合

sigma_sr_age = sigma_ratio_1s / abs(slope_dSr_dt)   # 得到 Sr 年龄 1σ (Myr)

print("示例 Sr 年龄 1σ ≈ %.2f Myr" % sigma_sr_age)

# 2.3 组装整条剖面的 σ 数组
sigma_ties = np.zeros_like(t_ties, dtype=float)

for i, ttype in enumerate(tie_types):
    if ttype == 'mag':
        sigma_ties[i] = sigma_mag_default
    elif ttype == 'bio':
        sigma_ties[i] = sigma_bio_default
    elif ttype == 'sr':
        sigma_ties[i] = sigma_sr_age
    else:
        raise ValueError("Unknown tie type: %s" % ttype)

# =================================================================
# 3. 沉积段分段：告诉模型哪里可以插值，哪里是 hiatus
# =================================================================
# 示例方案（你可根据最终年龄框架调整）：
#   段 0：4.8–12.8 Ma       (123.1–247.2 mbsf)   → H1 之前
#   段 1：16.4–22.5 Ma      (253.6–267.0 mbsf)   → H1 之后, H2 之前
#   段 2：23.1–25.4 Ma      (267.0–279.0 mbsf)   → H2 之后, H3 之前
#   段 3：26.9–34.4 Ma      (279.0–340.0 mbsf)   → H3 之后
segment_ids = np.array([
    0,  # 4.8
    0,  # 9.8
    0,  # 10.5
    0,  # 11.7
    0,  # 12.8  (H1 top)
    1,  # 16.4  (H1 base)
    1,  # 22.5  (H2 top)
    2,  # 23.1  (H2 base)
    2,  # 25.4  (H3 top)
    3,  # 26.9  (H3 base)
    3   # 34.4
])

# =================================================================
# 4. Monte Carlo 主体：按沉积段分开做线性插值
# =================================================================

# 深度网格
z_grid = np.linspace(z_ties.min(), z_ties.max(), 400)

# 保存每一次 realization 的年龄曲线
ages_mc = np.full((n_real, z_grid.size), np.nan)

unique_segments = np.unique(segment_ids)

for seg in unique_segments:
    # 取出该段内 tie points
    mask_seg = (segment_ids == seg)
    z_seg = z_ties[mask_seg]
    t_seg = t_ties[mask_seg]
    s_seg = sigma_ties[mask_seg]

    # 与该段对应的 z_grid 区间
    mask_grid = (z_grid >= z_seg.min()) & (z_grid <= z_seg.max())

    # 对该段进行 Monte Carlo
    for k in range(n_real):
        # (1) 在每个 tie point 上按 N(t_i, σ_i) 生成一个随机年龄
        t_sample = np.random.normal(loc=t_seg, scale=s_seg)

        # (2) 在该段深度范围内做一次线性插值
        ages_mc[k, mask_grid] = np.interp(z_grid[mask_grid], z_seg, t_sample)

# =================================================================
# 5. 统计平均年龄和标准差（忽略 NaN → hiatus 区间）
# =================================================================
mean_age = np.nanmean(ages_mc, axis=0)
std_age  = np.nanstd(ages_mc, axis=0)

age_1s_low  = mean_age - std_age
age_1s_high = mean_age + std_age
age_2s_low  = mean_age - 2 * std_age
age_2s_high = mean_age + 2 * std_age

# =================================================================
# 6. 作图：平均年龄 + Monte Carlo 误差带
# =================================================================
fig, ax = plt.subplots(figsize=(5, 6))

# 平均年龄–深度曲线
ax.plot(mean_age, z_grid, label='Mean age model', linewidth=1.5)

# 95% 置信带 (±2σ)
ax.fill_betweenx(z_grid, age_2s_low, age_2s_high,
                 alpha=0.3, label='95% envelope (±2σ)')

# 可选：画 1σ 带
# ax.fill_betweenx(z_grid, age_1s_low, age_1s_high,
#                  alpha=0.4, label='68% envelope (±1σ)')

# 标出三个位于 hiatus 边界的深度（可根据你的最终判断调整）
for d in [247.2, 253.6, 267.0, 279.0]:
    ax.axhline(d, color='k', linestyle='--', linewidth=0.5)

ax.set_xlabel('Age (Ma)')
ax.set_ylabel('Depth (mbsf)')
ax.invert_yaxis()
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()

plt.tight_layout()
plt.show()
import pandas as pd

# 如果你还没显式算 age_2s_low / age_2s_high，就用 std_age 现算
age_2s_low  = mean_age - 2.0 * std_age
age_2s_high = mean_age + 2.0 * std_age

# 组装输出表
df_out = pd.DataFrame({
    "Depth_mbsf": z_grid,
    "Age_mean_Ma": mean_age,
    "Age_95_low_Ma": age_2s_low,     # mean - 2σ
    "Age_95_high_Ma": age_2s_high,   # mean + 2σ
    "Age_1sigma_Ma": std_age         # 可选：把1σ也一起输出，方便复用
})

# 可选：保留小数位（按需调整）
df_out = df_out.round({
    "Depth_mbsf": 3,
    "Age_mean_Ma": 4,
    "Age_95_low_Ma": 4,
    "Age_95_high_Ma": 4,
    "Age_1sigma_Ma": 4
})

# 输出到 Excel（路径按需修改）
out_path = r"E:\Ocean\U1516_age_model_95CI.xlsx"
df_out.to_excel(out_path, index=False, sheet_name="MC_95CI")

print(f"Excel saved: {out_path}")
