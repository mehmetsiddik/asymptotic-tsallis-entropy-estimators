# Asymptotic Behaviour of Tsallis and Shannon Entropy Estimators

This repository contains the plotting code and figure outputs used to examine the finite-sample and asymptotic behaviour of Tsallis and Shannon entropy estimators.

The repository is organised so that the figures can be reviewed directly and regenerated from simulation results when the required data table is available.

## Repository structure

```text
.
├── Asymptotic_Tsallis_GitHub.ipynb
├── requirements.txt
├── src/
│   └── plot_asymptotic_tsallis.py
├── data/
│   └── README.md
└── figures/
    ├── variance_convergence.png
    └── asymptotic_normality_q1.png
```

## Figures

The `figures/` folder contains the exported versions of the two main outputs:

1. `variance_convergence.png` — variance decay of the entropy estimators across sample sizes, dimensions, and reference distributions.
2. `asymptotic_normality_q1.png` — standardized convergence patterns compared with the standard normal reference.

## Data format

The plotting functions expect a tabular simulation output with the following columns:

```text
dist, d, estimator, n, var, z
```

where:

- `dist` denotes the reference distribution, such as `GG_light`, `GG_heavy`, or `Student`;
- `d` is the dimension;
- `estimator` is either `Tsallis` or `Shannon`;
- `n` is the sample size;
- `var` is the empirical variance estimate;
- `z` is the standardized score used in the asymptotic normality plot.

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
```

To regenerate the figures from a CSV file:

```bash
python src/plot_asymptotic_tsallis.py --input data/asymptotic_tsallis_results.csv --output figures
```

Alternatively, open the notebook:

```text
Asymptotic_Tsallis_GitHub.ipynb
```

and run the cells after loading the simulation results into a DataFrame named `df`.

## Notes

The notebook was cleaned for repository use by keeping the final plotting workflow, consolidating repeated imports, and placing exported figures in a separate `figures/` directory.
