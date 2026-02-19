# UTA Project — Learning Additive Value Functions for Multi-Criteria Decision Aiding

This repository provides an implementation of **UTA-based preference learning models** for multi-criteria decision aiding (MCDA), with a focus on **monotone and non-monotone additive value functions** and experimental benchmarking.

The project is developed in a research-oriented setting and serves as a basis for comparing classical and recent preference learning approaches based on additive value models.

---

## 📌 Project Overview

The goal of this project is to:
- implement **UTA-style additive value function learning**,
- study the impact of **monotonicity vs non-monotonicity** assumptions on criteria,
- provide a clean experimental pipeline for **benchmarking preference learning methods**.

The implementation follows the MCDA literature, starting from classical UTA models and extending them toward more flexible formulations.

---

## 🧠 Methods Implemented

- **UTA (1982)**  
  Learning additive value functions from preference information.

- **Monotone UTA**  
  Marginal value functions constrained to be monotone.

- **Non-monotone UTA**  
  Marginal value functions without monotonicity constraints, allowing richer preference shapes.

The models are formulated as **linear optimization problems** with piecewise-linear marginal value functions.

---

## ⚙️ Model Structure

The global value of an alternative \( a \) is defined as:

\[
U(a) = \sum_{j=1}^{m} u_j(g_j(a))
\]

where:
- \( g_j(a) \) is the performance of alternative \( a \) on criterion \( j \),
- \( u_j(\cdot) \) is a marginal value function, approximated by piecewise-linear segments.

Breakpoints are fixed a priori, and the optimization learns the marginal utility values at these breakpoints.

---

## 📥 Inputs

Depending on the experiment, the model takes as input:
- a set of alternatives described by multiple criteria,
- preference information derived from a ground-truth value function (synthetic setting).

---

## 📤 Outputs

The learning process produces:
- learned marginal value functions for each criterion,
- global utility scores for alternatives,
- experimental results stored in CSV format for further analysis.

---

## 🧪 Experiments

Experiments are conducted to compare:
- monotone vs non-monotone models,
- approximation quality of the learned value functions,
- robustness to modeling assumptions.

All experiments and visualizations are handled in the provided Jupyter notebook.

---

## 📁 Repository Structure

```text
UTA_project/
│
├── uta_core.py                  # Core implementation of UTA models and optimization
├── uta.ipynb                    # Experimental notebook (learning, evaluation, plots)
├── uta_experiment__results.csv  # Saved experimental results
└── README.md
