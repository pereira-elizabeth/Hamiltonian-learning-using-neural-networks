
# Hamiltonian Learning using Neural Networks

## ML Fidelity vs Noise — Skills Demo

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/pereira-elizabeth/hamiltonian-ml)
![Code Size](https://img.shields.io/github/languages/code-size/pereira-elizabeth/hamiltonian-ml)
<!-- CI badge (works after you add .github/workflows/tests.yml) -->
<!-- ![Build](https://github.com/pereira-elizabeth/hamiltonian-ml/actions/workflows/tests.yml/badge.svg) -->
This project shows how a neural network can learn the “fingerprint” of a material and predict how it behaves when we change its properties.

This repository contains a **machine learning demo** in condensed matter physics:  
predicting real onsite energies of a 1D tight-binding chain and comparing **spectral densities (local density of states, LDOS) for predicted vs true Hamiltonian parameters**.

The Jupyter notebook includes:
- data generation functions for training and testing,
- a simple neural network model (Keras/TensorFlow),
- a framework to study robustness of predictions against noise using **fidelity** as a metric,
- inline comments throughout to make the workflow easy to follow.

---

## 📖 Contents
- `real_onsite_energy_learning.ipynb` – main Jupyter notebook
- `results/` – saved plots used in this README
- `requirements.txt` – dependencies for reproducibility
- `.gitignore` – ignores caches, data, checkpoints

---

## ⚡ Demo Results

### Predicted vs True Onsite Energies (no noise)
Scatter plot with regression line.
![Prediction vs True](results/pred_vs_true_jointplot_test.png)

---

### LDOS Comparisons
Local density of states (LDOS) for a single test sample:

- Without noise  
  ![LDOS comparison](results/ldos_true_pred1.png)

- With noise strength = 0.2, **with overfitting**  
  ![LDOS comparison](results/ldos_true_pred_noisy_with_overfitting.png)

- With noise strength = 0.2, **without overfitting**  
  ![LDOS comparison](results/ldos_true_pred_noisy_nooverfitting.png)

---

### Fidelity vs Noise

- With overfitting  
  ![Fidelity vs Noise](results/fidelity_vs_noise_with_overfitting.png)

- Without overfitting (improved network)  
  ![Fidelity vs Noise](results/fidelity_vs_noise_nooverfitting.png)

---
## Tests
Run the test suite:

```bash
pytest -q
```
---
## 🚀 Quickstart

Clone this repository and install dependencies:

```bash
git clone https://github.com/pereira-elizabeth/Hamiltonian-learning-using-neural-networks.git
cd Hamiltonian-learning-using-neural-networks
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
