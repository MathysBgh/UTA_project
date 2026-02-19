# UTA Project — Preference Learning with Additive Value Models

This repository contains a research-oriented implementation of **UTA-based preference learning methods** for multi-criteria decision aiding (MCDA).

The project focuses on learning **additive value functions** from preference information and serves as a basis for comparing classical and recent approaches in the MCDA literature.

---

## 🎯 Project Goal

The main objective is to study how different assumptions on criteria preferences impact the learned decision model, in particular:

- monotone vs non-monotone criteria,
- classical UTA vs more recent preference learning extensions,
- robustness and interpretability of learned models.

This work is part of a broader experimental benchmark in a research / MSc context.

---

## 📚 Methods and References

The project is grounded in the following key papers:

- **UTA (Jacquet-Lagrèze & Siskos, 1982)**  
  Classical preference disaggregation method learning additive value functions from preference information.

- **Ghaderi et al. (EJOR, 2017)**  
  Extension of UTA allowing **non-monotonic marginal value functions**, with an explicit trade-off between model complexity and discrimination power.

More recent methods (e.g. sorting-based and probabilistic extensions) are considered as potential benchmarks on top of this core implementation.

---

## 🧠 What the Model Does (High-Level)

- Takes alternatives described by multiple criteria.
- Learns marginal value functions for each criterion.
- Aggregates them into a global score representing preferences.
- Allows both monotone and non-monotone preference structures.

The focus is on **interpretability and controlled model complexity**, rather than black-box prediction.

---

## 🧪 Experiments

Experiments are conducted on synthetic data to:
- compare monotone and non-monotone models,
- analyze robustness to modeling assumptions,
- evaluate approximation quality of learned preferences.

Results are saved for reproducibility and analysis.

---

## 📁 Repository Structure

```text
UTA_project/
│
├── uta_core.py                  # Core UTA implementation
├── uta.ipynb                    # Experiments and analysis
├── uta_experiment__results.csv  # Experimental results
└── README.md
