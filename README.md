# Hamiltonian-learning-using-neural-networks

# ML Fidelity vs Noise — Skills Demo

This repository contains a small **machine learning demo** in condensed matter physics:  
predicting real onsite energies of a 1D chain and comparing **predicted vs true spectral densities**.  
The jupyter notebook includes the necessary functions to make the training and testing data of the neural network, the neural network, and also a code layout to study the robustness of neural netowrk, it is studied using the quantity called fidelity. The jupyter file has comments throughout the file for easy explanation.
---

## 📖 Contents
- `real_onsite_enery_learning.ipynb` – Jupyter notebook with full workflow
- `results/` – saved plots used in this README
- `requirements.txt` – dependencies for reproducibility
- `.gitignore` – ignores caches, data, checkpoints

---

## ⚡ Demo Results

### Fidelity vs Noise
Shows how prediction fidelity drops as input noise increases when there is overfitting of training data.
![Fidelity vs Noise](results/fidelityvsnoise_with_overfitting.png)

Shows how prediction fidelity drops as input noise increases when we remove overfitting of training data by using a different neural network.
![Fidelity vs Noise](results/fidelityvsnoise_nooverfitting.png)

### LDOS Comparison
True vs predicted local density of states (LDOS) for one test sample without any noise.
![LDOS comparison](results/ldos_true_pred1.png)

True vs predicted local density of states (LDOS) for one test sample with noise of strength $0.2$ with overfitting.
![LDOS comparison](results/ldos_true_pred_noisy_with_overfitting.png)

True vs predicted local density of states (LDOS) for one test sample with noise of strength $0.2$ without overfitting.
![LDOS comparison](results/ldos_true_pred_noisy_nooverfitting.png)

### Predicted vs True Onsite Energies
Scatter plot with regression line.
![Prediction vs True](results/pred_vs_true_jointplot_test.png)

---

## 🚀 Quickstart

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/ml-fidelity-demo.git
cd ml-fidelity-demo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
