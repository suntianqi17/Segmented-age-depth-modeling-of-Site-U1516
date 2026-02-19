# Segmented Age–Depth Modeling of IODP Site U1516

This repository contains Python scripts used to construct a segmented, weighted age–depth model for IODP Site U1516 (Expedition 369, Mentelle Basin).

## Overview

The model integrates:
- Shipboard magnetostratigraphic and biostratigraphic tie points
- Sr-isotope stratigraphic constraints
- Segmented weighted linear regressions
- Monte Carlo simulations for age uncertainty estimation

A weighting ratio of 100:1 (shipboard vs. Sr-isotope ages) was applied to ensure the age framework is primarily anchored to high-confidence stratigraphic tie points.

## Methods

The stratigraphic sequence is partitioned into depth segments based on:
1. Independent stratigraphic markers
2. Breaks in Sr-isotope trends

Within each segment, a weighted least-squares regression (Age = a × Depth + b) is applied.

Uncertainty propagation is performed using Monte Carlo simulations.

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
