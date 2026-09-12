# Azolla Mechanistic Biofilter Model V2.1

Reproducibility repository for the manuscript **An Identifiability-First Mechanistic Framework for Active Azolla Biofilters: Coupling Physiology, Mass Transfer and Model-Form Uncertainty**.

## Scope
This repository supports a literature-constrained mechanistic screening model for active *Azolla* biofilters used as a case study in indoor CO2 control. The work focuses on structural dependency, numerical verification, global sensitivity, practical identifiability, and hydraulic model-form uncertainty. It does **not** claim experimental validation of a fabricated Azolla device.

## Core files
- `notebooks/Azolla_Mechanistic_V2_1_Final_Colab.ipynb` — primary V2.1 mechanistic model notebook.
- `notebooks/Azolla_Computational_Experiments_Colab.ipynb` — supporting computational experiments.
- `notebooks/Azolla_Python_CFD_Reproducible.ipynb` — reproducible CFD/supporting notebook retained for provenance.
- `scripts/sobol_converged_2048.py` — nested Sobol analysis to N=2048 with bootstrap confidence intervals.
- `scripts/identifiability_diagnostic.py` — local sensitivity correlation and SVD diagnostic.
- `results/sobol_convergence_128_2048.csv` — sensitivity indices and 95% CI half-widths.
- `results/normalized_local_sensitivity_matrix.csv` — local normalized sensitivity signatures.
- `results/local_sensitivity_parameter_correlation.csv` — parameter-sensitivity correlation matrix.
- `results/svd_singular_values.csv` — singular-value spectrum.
- `results/sobol_raw_2048.npz` — sampled inputs and model outputs used for the converged Sobol analysis.

## Reproducibility
The converged global sensitivity analysis uses a scrambled Sobol low-discrepancy sequence with nested base sample sizes N = 128, 256, 512, 1024 and 2048, together with 1000 bootstrap resamples for confidence intervals. The local practical-identifiability diagnostic uses ±1% central finite differences around the reference parameter vector and the 30-min room CO2 trajectory at 30-s resolution.

## Software environment
Python 3 with NumPy, pandas, SciPy, Matplotlib and Numba. SALib is required only by the legacy/supporting workflow where applicable. See `requirements.txt`.

## Citation
Please cite the associated manuscript and the archived repository release once the DOI is minted. Citation metadata are provided in `CITATION.cff`.

## Permanent archive / DOI
GitHub repository: https://github.com/virayupr/azolla-mechanistic-biofilter-model

A versioned archival DOI will be added here after the repository is archived with a DOI-minting service such as Zenodo.
