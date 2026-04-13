# Interpretable Preference Learning: From UTA to Zhang Sorting

This repository contains the code, notebooks, experimental outputs, and communication material for a research project on **interpretable additive preference learning** in multi-criteria decision aiding (MCDA).

The project started with **classical UTA**, was extended to **non-monotone UTA** following Ghaderi et al., and now also includes a **Zhang-based representative sorting model**. The common goal is to learn additive models from partial supervision while preserving:

- interpretability
- optimization-based formulation
- controlled synthetic evaluation
- meaningful comparison across model families

## Project Question

> How far can we improve preference realism beyond monotonicity while keeping additive models interpretable, optimization-based, and experimentally comparable?

In practical terms, the repository studies two related tasks:

- **ranking** from pairwise preferences
- **sorting** from reference assignments to ordered categories

## At a Glance

| Method | Supervision | Learned object | Main use |
| --- | --- | --- | --- |
| Classical UTA | Pairwise preferences | Monotone additive utility | Ranking reconstruction |
| Ghaderi et al. (2017) | Pairwise preferences | Non-monotone additive utility | Ranking with sweet spots and U-shaped criteria |
| Zhang et al. (2025) | Reference assignments | Additive sorting model with thresholds | Interpretable sorting |

## Why This Project Matters

Classical additive preference models usually assume that each criterion is monotone: more is always better, or less is always better. This is often too restrictive in realistic decision problems.

Typical counterexamples include:

- workload
- room temperature
- sweetness
- class size
- city size

For these criteria, the preferred value is often an **intermediate range**, not an extreme. This project therefore evaluates not only predictive performance, but also whether the learned model recovers the **true shape of preferences**.

## Main Contributions

- A synthetic benchmark with known ground-truth marginals.
- A clean implementation of **classical UTA**.
- A working implementation of **non-monotone UTA** inspired by Ghaderi et al.
- A complete **Zhang representative sorting pipeline** with consistency checking and repair.
- Dedicated notebooks for:
  - UTA analysis
  - Zhang analysis
  - first UTA vs Zhang comparison
- Poster and defense material generated from the experimental outputs.

## Visual Snapshot

<p align="center">
  <img src="poster_assets/uta_highlights.png" width="48%" alt="UTA benchmark highlights" />
  <img src="poster_assets/zhang_highlights.png" width="48%" alt="Zhang benchmark highlights" />
</p>

<p align="center">
  <img src="poster_assets/uta_vs_zhang_headtohead.png" width="60%" alt="UTA vs Zhang comparison" />
</p>

## Implemented Models

### 1. Classical UTA

Implemented components:

- additive value functions with monotone marginal utilities
- fitting from pairwise preferences
- synthetic benchmark generation
- evaluation of predictive accuracy, rank agreement, structural recovery, and runtime

### 2. Non-monotone UTA (Ghaderi et al.)

Implemented components:

- relaxed monotonicity
- flexible marginal shapes
- slope-variation regularization
- structural detection metrics such as `type_acc`

### 3. Zhang Representative Sorting

Implemented components:

- additive sorting model with thresholds
- lexicographic optimization with two approaches:
  - `A1`: minimize complexity, then maximize separation
  - `A2`: maximize separation, then minimize complexity
- consistency checking and optional assignment repair
- sorting metrics such as:
  - `acc_test`
  - `mono_recovery_acc`
  - `epsilon`
  - `fit_ok`

## Experimental Design

All experiments are run on **synthetic data** so that the ground-truth marginal functions are known. This is important because it allows us to evaluate both:

- predictive accuracy
- structural recovery of the learned marginals

Typical factors varied in the benchmark:

- number of criteria `m`
- number of breakpoints `L`
- number of non-monotone criteria `k_non_mono`
- peak type (`central`, `extreme`)
- noise level
- supervision budget:
  - number of pairwise comparisons for UTA
  - reference ratio for Zhang

## Key Results

### UTA side

From the current main CSV:

| Method | Mean `acc_test` | Mean `type_acc` | Mean `cpu_sec` |
| --- | ---: | ---: | ---: |
| `A_mono` | `0.8229` | `0.6719` | `0.1189` |
| `B_nonmono` | `0.8362` | `0.7047` | `0.1157` |

Main takeaway:

- the non-monotone extension improves predictive fidelity
- it also improves structural recovery
- runtime remains in the same range as the monotone baseline

### Zhang side

From the current main CSV:

| Method | Mean `acc_test` | Mean `mono_recovery_acc` | Mean `cpu_sec` | Mean `epsilon` |
| --- | ---: | ---: | ---: | ---: |
| `A1` | `0.8616` | `0.6382` | `0.0836` | `0.0465` |
| `A2` | `0.8278` | `0.5093` | `0.0813` | `0.2637` |

Main takeaway:

- `A1` is the best choice for predictive accuracy
- `A2` gives stronger category separation
- Zhang exposes a clear accuracy/separation trade-off

### First UTA vs Zhang bridge

The comparison notebook currently provides an exploratory matched-condition bridge:

- mean delta on matched settings: `+0.0232` for Zhang
- median delta: `+0.0270`
- Zhang win rate: `86.2%`

Important caveat:

- this is **not** a strict apples-to-apples comparison yet
- UTA predicts rankings from pairwise labels
- Zhang predicts categories from reference assignments

## Repository Guide

### Core implementations

- `uta_core.py` - core utilities for UTA-style synthetic generation and inference
- `zhang_core.py` - implementation of the Zhang representative sorting model

### Main notebooks

- `uta.ipynb` - main notebook for the UTA / non-monotone UTA study
- `zhang_notebook_runner.ipynb` - first runnable Zhang notebook and sanity checks
- `zhang_protocol.ipynb` - large-scale Zhang experimental protocol with CSV export
- `zhang_results_viz.ipynb` - dedicated Zhang visualization notebook
- `uta_vs_zhang_comparison.ipynb` - first bridge notebook for cross-family comparison

### Experimental outputs

- `uta_experiment__results.csv` - main UTA benchmark results
- `zhang_experiment_results.csv` - main Zhang benchmark results
- `zhang_experiment_results_new.csv` - secondary Zhang run artifact

### Scripts and utilities

- `generate_poster_assets.py` - generates poster-ready figures from CSV results
- `build_zhang_defense_update.py` - creates a concise updated presentation including Zhang
- `build_zhang_update_same_layout.py` - updates the original defense deck while preserving its layout

### Communication material

- `Lab Project Defense.pptx` / `Lab Project Defense.pdf`
- `poster_baposter.tex`
- `poster_a1.tex`
- `poster_template_style.tex`
- `poster_assets/`

## Recommended Reading Order

If someone opens the repository for the first time, the cleanest path is:

1. Start with `uta.ipynb` to understand the benchmark and the monotone vs non-monotone UTA comparison.
2. Open `zhang_notebook_runner.ipynb` for a first Zhang run and sanity checks.
3. Run or inspect `zhang_protocol.ipynb` for the large-scale Zhang protocol.
4. Use `zhang_results_viz.ipynb` for Zhang-only figures.
5. Finish with `uta_vs_zhang_comparison.ipynb` for the first cross-method bridge.

## Requirements

Main dependencies:

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `gurobipy`

Optional dependencies:

- `python-pptx` for presentation generation
- `Pillow` for image sizing in generated slides

Important note:

- optimization-based parts require a working **Gurobi** installation and a valid license
- some larger Zhang runs can hit the limits of a size-limited academic license

## Reproducibility Notes

- The main scientific entry points are the Python core files, notebooks, and final CSV outputs.
- Some files in the repository are exploratory, historical, or presentation-oriented; they remain useful, but they are not the primary scientific source of truth.
- The project mixes research code, benchmark notebooks, and communication material in the same repository; the `Repository Guide` section above is the quickest way to navigate it cleanly.

## Current Status

Current repository status:

- classical UTA implemented and benchmarked
- non-monotone UTA implemented and benchmarked
- Zhang representative sorting implemented
- Zhang protocol notebook completed
- Zhang visualization notebook completed
- first UTA vs Zhang bridge completed
- poster and defense material updated from the latest experimental results

## Authors

- Mathys Bagnah
- Sydney Nzunguli

---

If you are reading this repository as a supervisor, reviewer, or collaborator, the best entry points are:

- `uta.ipynb`
- `zhang_protocol.ipynb`
- `zhang_results_viz.ipynb`
- `uta_vs_zhang_comparison.ipynb`
