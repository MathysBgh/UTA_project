# Interpretable Preference Learning: From UTA to Zhang Sorting

This repository contains the tracked code and main analysis files for a project on **interpretable additive preference learning** in multi-criteria decision aiding.

The work started with **classical UTA**, was extended to a **non-monotone UTA** setting inspired by Ghaderi et al., and then connected to a **Zhang-style representative sorting** model.

## Project Idea

The main question behind the project is simple:

> how far can we relax monotonicity while keeping the model interpretable and optimization-based?

In practice, the repository touches two related tasks:

- **ranking** from pairwise preferences
- **sorting** from reference assignments

## Methods Covered

| Method | Input | Output |
| --- | --- | --- |
| UTA | pairwise preferences | monotone additive utility |
| Ghaderi-style extension | pairwise preferences | non-monotone additive utility |
| Zhang-style approach | reference assignments | additive sorting model with thresholds |


### Core code

- `uta_core.py`  
  Core utilities for UTA-style generation, scaling, interpolation, and inference.

- `zhang_core.py`  
  Core implementation of the Zhang representative sorting model.

### Main notebooks

- `uta.ipynb`  
  Main notebook for the UTA and non-monotone UTA study.

- `zhang_notebook_runner.ipynb`  
  Main runnable notebook for Zhang experiments and quick checks.

- `uta_vs_zhang_comparison.ipynb`  
  First comparison notebook connecting the UTA side and the Zhang side.

### Experimental outputs

- `uta_experiment__results.csv`  
  Main result file for the UTA benchmark.

- `zhang_experiment_results_new.csv`  
  Main tracked result file for the Zhang side.

### Documentation

- `README.md`  
  Project overview and repository guide.

## Suggested Reading Order

If you want to understand the repository quickly, a good order is:

1. `uta.ipynb`
4. `zhang_notebook_runner.ipynb`
5. `uta_vs_zhang_comparison.ipynb`

## Main Dependencies

The project mainly relies on:

- Python 3
- `numpy`
- `pandas`
- `matplotlib`
- `gurobipy`

## Notes

- The optimization-based parts require a working **Gurobi** installation and a valid license.
- The repository is a mix of core code, notebooks, and result files.
- Some additional local working files may exist outside the tracked Git contents

## Authors

- Mathys Bagnah
- Sydney Nzunguli
