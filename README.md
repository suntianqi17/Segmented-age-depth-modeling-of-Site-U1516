# U1516 Age Model

Python scripts used to construct the revised Oligocene–Miocene
age-depth model and estimate chronological uncertainties for
IODP Site U1516, Mentelle Basin.

## Contents

- `Sr_age_calculation.py`: converts measured seawater 87Sr/86Sr
  ratios to numerical ages using the McArthur et al. (2025)
  seawater Sr-isotope calibration.

- `U1516_age_depth_model.py`: constructs the piecewise age-depth
  model using Sr-derived ages together with shipboard
  biostratigraphic and magnetostratigraphic age constraints.

- `U1516_age_uncertainty.py`: estimates age uncertainties using
  Monte Carlo simulations.

## Dependencies

- NumPy
- Pandas
- SciPy
- Matplotlib

## Data Availability

Stratigraphic data are derived from IODP Expedition 369 (Site U1516).
Dry bulk density data are available via the IODP LORE database.
GEBCO 2023 bathymetry is used for regional mapping.

## Citation

If you use this code, please cite the associated manuscript:

Sun et al. (2026), *Chemostratigraphic constraints on Oligocene–Miocene sedimentation and hiatus formation at IODP Site U1516*, Paleoceanography and Paleoclimatology.
