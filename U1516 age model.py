import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

#  ==============================================
# elsx读取数据，并进行可视化
#  ==============================================
# 1.1 读入 Excel
file_path = r"E:\data for python/original age control point & Sr age of samples.xlsx"  # 改成你本地路径
df = pd.read_excel(file_path)

# 1.2 安全地把 Top/Bottom 深度转成数值
def to_float(x):
    try:
        return float(x)
    except:
        return np.nan

df["Top"] = df["Top depth（mbsf）"].apply(to_float)
df["Bottom"] = df["Bottom depth（mbsf）"].apply(to_float)


col3 = df.columns[3]   # 自动获取第三列名，例如 "Age control point"

# 定义一个判断是否为空的函数（处理 NaN、"/"、空格）
def not_empty(x):
    if pd.isna(x):
        return False
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x == "/" or x == "／":
            return False
    return True

# 提取第三列不为空的行
df_top_bottom = df[df[col3].apply(not_empty)]

# 查看结果
print(df_top_bottom)

# 1.3 构造“控制点深度”：
#    - 若你在表里已经有一列修订后的单一深度（例如叫 "Depth_ctrl"），优先使用；
#    - 否则：Top 和 Bottom 都有 → 取中值；只有一边 → 用那一边。
if "Depth_ctrl" in df.columns:
    depth_ctrl = df["Depth_ctrl"].apply(to_float)
else:
    depth_ctrl = df["Top"].where(df["Top"].notna(), df["Bottom"])
    # Top 和 Bottom 都有的，取中值
    both_mask = df["Top"].notna() & df["Bottom"].notna()
    depth_ctrl[both_mask] = (df.loc[both_mask, "Top"] + df.loc[both_mask, "Bottom"]) / 2

age_ctrl = pd.to_numeric(df["Age（Ma）」"], errors="coerce") if "Age（Ma）」" in df.columns else pd.to_numeric(df["Age（Ma）"], errors="coerce")

ctrl = pd.DataFrame({"Depth": depth_ctrl, "Age": age_ctrl}).dropna()

depth_ctrl = ctrl["Depth"].to_numpy()
age_ctrl = ctrl["Age"].to_numpy()

print("控制点数量:", len(depth_ctrl))

# 1.4 Sr 样品深度和 Sr 反演年龄
depth_sam = pd.to_numeric(df["Depth_m"], errors="coerce").dropna().to_numpy()
age_sam = pd.to_numeric(df["Sr_age_Ma"], errors="coerce").dropna().to_numpy()

print("Sr 样品数量:", len(depth_sam))
# 2 把Sr-时间与原始年龄控制点放在同一张图上
plt.figure()
plt.scatter(age_ctrl, depth_ctrl, label="Age control points")  # 控制点
plt.scatter(age_sam, depth_sam, s=15, label="Sr ages")         # Sr 样品
plt.gca().invert_yaxis()
plt.xlabel("Age (Ma)")
plt.ylabel("Depth (m)")
plt.legend()
plt.title("Revised age control points + Sr ages")
plt.tight_layout()
plt.show()


# ===============================================
# 处理数据
# ===============================================
# 3.1 手动设置分段边界（可根据图反复调整）
# 例如： [120, 200, 250, 290, 340]
segment_bounds = [123.1, 220.7, 224, 245.3, 247.2, 253.6, 267, 279, 340]  # 单位 m

# 确保按从浅到深排序
segment_bounds = sorted(segment_bounds)
# 3.2 构建联合数据（控制点 + Sr 点）
depth_all = np.concatenate([depth_ctrl, depth_sam])
age_all = np.concatenate([age_ctrl, age_sam])



weights_ctrl = np.full_like(age_ctrl, 100.0, dtype=float)  # 控制点高权重
weights_sam = np.full_like(age_sam, 1.0, dtype=float)  # Sr 样品中等权重
weights_all = np.concatenate([weights_ctrl, weights_sam])

# 按深度排序
idx = np.argsort(depth_all)
depth_all_sorted = depth_all[idx]
age_all_sorted = age_all[idx]
weights_sorted = weights_all[idx]

# 3.3 为每一段拟合局部样条
age_model_depth = []
age_model_age = []

segment_models = []  # 保存每段的样条模型，方便之后用
#
# for i in range(len(segment_bounds) - 1):
#     z_min = segment_bounds[i]
#     z_max = segment_bounds[i + 1]
#
#     mask_seg = (depth_all_sorted >= z_min) & (depth_all_sorted <= z_max)
#     dep_seg = depth_all_sorted[mask_seg]
#     age_seg = age_all_sorted[mask_seg]
#     w_seg = weights_sorted[mask_seg]
#
#     if len(dep_seg) < 4:
#         # 点太少，用简单线性插值
#         if len(dep_seg) >= 2:
#             spline_seg = UnivariateSpline(dep_seg, age_seg, w=w_seg, s=0.5, k=2)
#         else:
#             continue
#     else:
#         # 通常使用三次样条 + 小的平滑因子（避免振荡）
#         s_seg = len(dep_seg) * 0.01
#         spline_seg = UnivariateSpline(dep_seg, age_seg, w=w_seg, s=0.5, k=2)
#
#     # 在该段内生成一个网格
#     depth_grid_seg = np.linspace(z_min, z_max, 200)
#     age_grid_seg = spline_seg(depth_grid_seg)
#
#     age_model_depth.append(depth_grid_seg)
#     age_model_age.append(age_grid_seg)
#     segment_models.append((z_min, z_max, spline_seg))


# ================================================
# 分段线性拟合（无权重）
# ================================================
age_model_depth_polyfit = []
age_model_age_polyfit = []
segment_models_polyfit = []

for i in range(len(segment_bounds) - 1):
    z_min = segment_bounds[i]
    z_max = segment_bounds[i + 1]

    # 该分段的数据
    mask_seg = (depth_all_sorted >= z_min) & (depth_all_sorted <= z_max)
    dep_seg = depth_all_sorted[mask_seg]
    age_seg = age_all_sorted[mask_seg]
    w_seg = weights_sorted[mask_seg]     # 线性拟合本身不使用 w_seg，但保留

    # --------------------------
    # 线性拟合（核心修改）
    # --------------------------
    if len(dep_seg) >= 2:
        # 一阶多项式 = 直线拟合: age = a * depth + b
        a, b = np.polyfit(dep_seg, age_seg, 1)

        # 定义该段的拟合函数
        def linear_func(x, aa=a, bb=b):
            return aa * x + bb

        spline_seg = linear_func

    else:
        # 无法拟合（点数不足）
        continue

    # 生成该段的网格点
    depth_grid_seg = np.linspace(z_min, z_max, 200)
    age_grid_seg = spline_seg(depth_grid_seg)

    # 保存结果
    age_model_depth_polyfit.append(depth_grid_seg)
    age_model_age_polyfit.append(age_grid_seg)
    segment_models_polyfit.append((z_min, z_max, spline_seg))


# ================================================
# 分段线性拟合（有权重）
# ================================================
def weighted_linear_fit(x, y, w):
    """
    加权线性拟合： y = a*x + b
    x, y, w 为 numpy 数组（w 为点权重）
    返回：a, b
    """
    X = np.vstack([x, np.ones_like(x)]).T  # 设计矩阵
    W = np.diag(w)                         # 权重矩阵
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    a, b = beta[0], beta[1]
    return a, b


age_model_depth_WLS = []
age_model_age_WLS = []
segment_models_WLS = []

for i in range(len(segment_bounds) - 1):
    z_min = segment_bounds[i]
    z_max = segment_bounds[i + 1]

    # 该分段的数据
    mask_seg = (depth_all_sorted >= z_min) & (depth_all_sorted <= z_max)
    dep_seg = depth_all_sorted[mask_seg]
    age_seg = age_all_sorted[mask_seg]
    w_seg = weights_sorted[mask_seg]   # 每个点自己的权重

    # -----------------------------------
    #      ⭐ 加权线性拟合（核心更新）
    # -----------------------------------
    if len(dep_seg) >= 2:
        a, b = weighted_linear_fit(dep_seg, age_seg, w_seg)

        # 定义该段的拟合函数
        def linear_func(x, aa=a, bb=b):
            return aa * x + bb

        spline_seg = linear_func
    else:
        continue
    print(f'线段编号 {i} \t a: {a}, b: {b}')
    # 生成该段的网格点
    depth_grid_seg = np.linspace(z_min, z_max, 200)
    age_grid_seg = spline_seg(depth_grid_seg)

    # 保存结果
    age_model_depth_WLS.append(depth_grid_seg)
    age_model_age_WLS.append(age_grid_seg)
    segment_models_WLS.append((z_min, z_max, spline_seg))


# ================================================
# 绘图
# ================================================
fig, axs = plt.subplots(1, 2, figsize=(8, 8))


# ------------------------------------------------
# 绘制分段拟合直线
# ------------------------------------------------
ax = axs[0]

ax.scatter(age_ctrl, depth_ctrl, label="Age control points")
ax.scatter(age_sam, depth_sam, s=15, label="Sr ages")

for i_in in range(len(segment_models_polyfit)):
    _a = age_model_depth_polyfit[i_in]
    _b = age_model_age_polyfit[i_in]
    ax.plot(_b, _a, "-k", linewidth=0.5,
             label=f"Age-depth model {i_in}")

_top = df_top_bottom['Top depth（mbsf）'].to_numpy()
_bottom = df_top_bottom['Bottom depth（mbsf）'].to_numpy()
_age = df_top_bottom['Age（Ma）'].to_numpy()
for t, b, a in zip(_top, _bottom, _age):
    # 画一段从 (Top, Age) 到 (Bottom, Age) 的线
    ax.plot([a, a], [t, b], '--ro', )

ax.invert_yaxis()
ax.set_xlabel("Age (Ma)")
ax.set_ylabel("Depth (m)")
ax.legend()
ax.set_title("Segmented age-depth fit (polyfit)")
# ------------------------------------------------
# 绘制加权分段拟合直线
# ------------------------------------------------
ax = axs[1]

ax.scatter(age_ctrl, depth_ctrl, label="Age control points")
ax.scatter(age_sam, depth_sam, s=15, label="Sr ages")

for i_in in range(len(segment_models_WLS)):
    _a = age_model_depth_WLS[i_in]
    _b = age_model_age_WLS[i_in]
    ax.plot(_b, _a, "-k", linewidth=0.5,
             label=f"Age-depth model {i_in}")

_top = df_top_bottom['Top depth（mbsf）'].to_numpy()
_bottom = df_top_bottom['Bottom depth（mbsf）'].to_numpy()
_age = df_top_bottom['Age（Ma）'].to_numpy()
for t, b, a in zip(_top, _bottom, _age):
    # 画一段从 (Top, Age) 到 (Bottom, Age) 的线
    ax.plot([a, a], [t, b], '--ro', )

ax.invert_yaxis()
ax.set_xlabel("Age (Ma)")
ax.set_ylabel("Depth (m)")
ax.legend()
ax.set_title("Segmented age-depth fit (WLS)")

plt.tight_layout()
plt.show()





# # 3.3 拼接成整段模型
# age_model_depth = np.concatenate(age_model_depth)
# age_model_age = np.concatenate(age_model_age)
# plt.figure(figsize=(4, 8))
# plt.scatter(age_ctrl, depth_ctrl, label="Age control points")
# plt.scatter(age_sam, depth_sam, s=15, label="Sr ages")
# plt.plot(age_model_age, age_model_depth, label="Segmented age-depth model")
# plt.gca().invert_yaxis()
# plt.xlabel("Age (Ma)")
# plt.ylabel("Depth (m)")
# plt.legend()
# plt.title("Segmented age-depth fit (revised control points)")
# plt.tight_layout()
# plt.show()

