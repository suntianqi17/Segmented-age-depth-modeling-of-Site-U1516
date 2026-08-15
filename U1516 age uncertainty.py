import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


# ============================================================
# 0. Basic settings
# ============================================================

n_real = 2000
np.random.seed(42)


# ============================================================
# 1. Age-control points
# ============================================================
#
# Depths are the final adapted depths (CSF-A, mbsf)
# used in the revised age model.
# ============================================================

z_ties = np.array([
    123.1,   # C3n.2r
    220.7,   # C4Ar.3r
    224.0,   # B Discoaster neohamatus
    245.3,   # B Globoturborotalita nepenthes
    247.2,   # B Reticulofenestra pseudoumbilicus - H1 top
    253.6,   # 87Sr/86Sr age - H1 base
    267.0,   # B Globoquadrina dehiscens - H2 top
    267.0,   # B Sphenolithus delphix - H2 base
    279.0,   # T Chiasmolithus altus - H3 top
    279.0,   # T Paragloborotalia opima - H3 base
    340.0    # T Discoaster saipanensis
])


# ============================================================
# 2. Ages of control points
# ============================================================

t_ties = np.array([
    4.8,     # C3n.2r
    9.8,     # C4Ar.3r
    10.5,    # D. neohamatus
    11.7,    # G. nepenthes
    12.8,    # R. pseudoumbilicus
    16.4,    # Sr isotope age
    22.5,    # G. dehiscens
    23.1,    # S. delphix
    25.4,    # Ch. altus
    26.9,    # P. opima
    34.4     # D. saipanensis
])


# ============================================================
# 3. Type of each age-control point
# ============================================================

tie_types = np.array([
    'mag',   # 4.8
    'mag',   # 9.8
    'bio',   # 10.5
    'bio',   # 11.7
    'bio',   # 12.8
    'sr',    # 16.4
    'bio',   # 22.5
    'bio',   # 23.1
    'bio',   # 25.4
    'bio',   # 26.9
    'bio'    # 34.4
])


# ============================================================
# 4. Age uncertainties assigned to tie points
# ============================================================
#
# These values are used as sigma in the Gaussian
# Monte Carlo perturbation.
#
# NOTE:
# The magnetostratigraphic and biostratigraphic values
# should later be checked against their source/calibration.
#
# The Sr-derived age at 253.6 mbsf is assigned
# an age uncertainty of 0.10 Myr.
# ============================================================

sigma_mag_default = 0.05    # Myr
sigma_bio_default = 0.20    # Myr
sigma_sr_age = 0.10         # Myr


# Create uncertainty array
sigma_ties = np.zeros_like(t_ties, dtype=float)

for i, ttype in enumerate(tie_types):

    if ttype == 'mag':
        sigma_ties[i] = sigma_mag_default

    elif ttype == 'bio':
        sigma_ties[i] = sigma_bio_default

    elif ttype == 'sr':
        sigma_ties[i] = sigma_sr_age

    else:
        raise ValueError(
            f"Unknown tie-point type: {ttype}"
        )


# Print input data for checking
print("\n====================================================")
print("Age-control points used in Monte Carlo")
print("====================================================")

for z, age, ttype, sigma in zip(
    z_ties,
    t_ties,
    tie_types,
    sigma_ties
):

    print(
        f"Depth = {z:6.1f} mbsf | "
        f"Age = {age:5.1f} Ma | "
        f"Type = {ttype:4s} | "
        f"Age uncertainty = {sigma:.2f} Myr"
    )


# ============================================================
# 5. Define continuous sedimentation segments
# ============================================================
#
# Segment 0:
# 4.8–12.8 Ma
# 123.1–247.2 mbsf
#
# H1:
# 12.8–16.4 Ma
# 247.2–253.6 mbsf
#
# Segment 1:
# 16.4–22.5 Ma
# 253.6–267.0 mbsf
#
# H2:
# 22.5–23.1 Ma at ~267 mbsf
#
# Segment 2:
# 23.1–25.4 Ma
# 267.0–279.0 mbsf
#
# H3:
# 25.4–26.9 Ma at ~279 mbsf
#
# Segment 3:
# 26.9–34.4 Ma
# 279.0–340.0 mbsf
# ============================================================

segment_ids = np.array([
    0,   # 4.8
    0,   # 9.8
    0,   # 10.5
    0,   # 11.7
    0,   # 12.8

    1,   # 16.4
    1,   # 22.5

    2,   # 23.1
    2,   # 25.4

    3,   # 26.9
    3    # 34.4
])


# ============================================================
# 6. Create depth grid
# ============================================================

z_grid = np.linspace(
    z_ties.min(),
    z_ties.max(),
    400
)


# ============================================================
# 7. Array for Monte Carlo realizations
# ============================================================

ages_mc = np.full(
    (n_real, z_grid.size),
    np.nan
)


unique_segments = np.unique(
    segment_ids
)


# ============================================================
# 8. Monte Carlo simulations
# ============================================================
#
# Each sedimentary segment is treated independently.
#
# No interpolation is performed across hiatuses.
#
# For each realization:
#
#   1. Randomly perturb each tie-point age
#      according to a Gaussian distribution.
#
#   2. Linearly interpolate age between control points
#      within each continuous sedimentation interval.
# ============================================================

for seg in unique_segments:

    # Select tie points belonging to this segment
    mask_seg = (
        segment_ids == seg
    )

    z_seg = z_ties[
        mask_seg
    ]

    t_seg = t_ties[
        mask_seg
    ]

    s_seg = sigma_ties[
        mask_seg
    ]


    # Sort by depth for safety
    sort_idx = np.argsort(
        z_seg
    )

    z_seg = z_seg[
        sort_idx
    ]

    t_seg = t_seg[
        sort_idx
    ]

    s_seg = s_seg[
        sort_idx
    ]


    # Depth-grid interval corresponding to this segment
    mask_grid = (
        (z_grid >= z_seg.min())
        &
        (z_grid <= z_seg.max())
    )


    # Run Monte Carlo realizations
    for k in range(n_real):

        # Randomly perturb tie-point ages
        t_sample = np.random.normal(
            loc=t_seg,
            scale=s_seg
        )


        # Linear interpolation within this segment
        ages_mc[
            k,
            mask_grid
        ] = np.interp(
            z_grid[
                mask_grid
            ],
            z_seg,
            t_sample
        )


# ============================================================
# 9. Calculate mean age and uncertainty
# ============================================================
#
# Hiatus intervals contain only NaN values.
# Suppress warnings caused by all-NaN hiatus grid cells.
# ============================================================

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        category=RuntimeWarning
    )

    mean_age = np.nanmean(
        ages_mc,
        axis=0
    )

    std_age = np.nanstd(
        ages_mc,
        axis=0
    )


# ============================================================
# 10. Calculate 1-sigma and 2-sigma envelopes
# ============================================================

age_1s_low = (
    mean_age - std_age
)

age_1s_high = (
    mean_age + std_age
)


age_2s_low = (
    mean_age - 2.0 * std_age
)

age_2s_high = (
    mean_age + 2.0 * std_age
)


# ============================================================
# 11. Print overall age-model uncertainty
# ============================================================

valid_std = std_age[
    np.isfinite(std_age)
]


print("\n====================================================")
print("Monte Carlo age-model uncertainty")
print("====================================================")


if len(valid_std) > 0:

    print(
        f"Minimum 1sigma uncertainty = "
        f"{np.nanmin(valid_std):.3f} Myr"
    )

    print(
        f"Maximum 1sigma uncertainty = "
        f"{np.nanmax(valid_std):.3f} Myr"
    )

    print(
        f"Mean 1sigma uncertainty = "
        f"{np.nanmean(valid_std):.3f} Myr"
    )


# ============================================================
# 12. Print uncertainty near important depths
# ============================================================

important_depths = [
    123.1,
    220.7,
    247.2,
    253.6,
    267.0,
    279.0,
    340.0
]


print("\n====================================================")
print("Uncertainty near important stratigraphic depths")
print("====================================================")


for target_depth in important_depths:

    idx = np.argmin(
        np.abs(
            z_grid - target_depth
        )
    )

    print(
        f"Requested depth = {target_depth:6.1f} mbsf | "
        f"Grid depth = {z_grid[idx]:7.3f} mbsf | "
        f"Mean age = {mean_age[idx]:7.3f} Ma | "
        f"1sigma = {std_age[idx]:6.3f} Myr"
    )


# ============================================================
# 13. Plot final age model and uncertainty
# ============================================================

fig, ax = plt.subplots(
    figsize=(5, 7)
)


# Mean age-depth model
ax.plot(
    mean_age,
    z_grid,
    label="Mean age model",
    linewidth=1.5
)


# 95% envelope
ax.fill_betweenx(
    z_grid,
    age_2s_low,
    age_2s_high,
    alpha=0.3,
    label="95% envelope (±2σ)"
)


# Optional 1-sigma envelope
ax.fill_betweenx(
    z_grid,
    age_1s_low,
    age_1s_high,
    alpha=0.2,
    label="±1σ"
)


# Plot original age-control points
ax.scatter(
    t_ties,
    z_ties,
    s=25,
    label="Age-control points",
    zorder=5
)


# Hiatus boundaries
for d in [
    247.2,
    253.6,
    267.0,
    279.0
]:

    ax.axhline(
        d,
        linestyle="--",
        linewidth=0.5
    )


ax.set_xlabel(
    "Age (Ma)"
)

ax.set_ylabel(
    "Depth (mbsf)"
)

ax.invert_yaxis()

ax.grid(
    True,
    linestyle="--",
    alpha=0.4
)

ax.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 14. Plot age uncertainty versus depth
# ============================================================

fig, ax = plt.subplots(
    figsize=(4, 7)
)


ax.plot(
    std_age,
    z_grid
)


ax.set_xlabel(
    "Age uncertainty (Myr, 1σ)"
)

ax.set_ylabel(
    "Depth (mbsf)"
)

ax.invert_yaxis()

ax.grid(
    True,
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.show()


# ============================================================
# 15. Export results to Excel
# ============================================================

df_out = pd.DataFrame({

    "Depth_mbsf":
        z_grid,

    "Age_mean_Ma":
        mean_age,

    "Age_1sigma_Ma":
        std_age,

    "Age_1sigma_low_Ma":
        age_1s_low,

    "Age_1sigma_high_Ma":
        age_1s_high,

    "Age_95_low_Ma":
        age_2s_low,

    "Age_95_high_Ma":
        age_2s_high
})


# Round values
df_out = df_out.round({

    "Depth_mbsf":
        3,

    "Age_mean_Ma":
        4,

    "Age_1sigma_Ma":
        4,

    "Age_1sigma_low_Ma":
        4,

    "Age_1sigma_high_Ma":
        4,

    "Age_95_low_Ma":
        4,

    "Age_95_high_Ma":
        4
})


# ============================================================
# 16. Export tie-point uncertainty table as another worksheet
# ============================================================

df_ties = pd.DataFrame({

    "Depth_mbsf":
        z_ties,

    "Age_Ma":
        t_ties,

    "Type":
        tie_types,

    "Age_uncertainty_Myr":
        sigma_ties,

    "Segment_ID":
        segment_ids
})


# ============================================================
# 17. Save Excel workbook
# ============================================================

out_path = r"E:\Ocean\U1516_age_model_MC_uncertainty.xlsx"


with pd.ExcelWriter(
    out_path
) as writer:

    df_out.to_excel(
        writer,
        sheet_name="MC_age_model",
        index=False
    )

    df_ties.to_excel(
        writer,
        sheet_name="Tie_points",
        index=False
    )


print(
    "\n===================================================="
)

print(
    f"Excel saved:\n{out_path}"
)

print(
    "===================================================="
)
