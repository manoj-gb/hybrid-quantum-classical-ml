# Hybrid Quantum-Classical Machine Learning

> Benchmarking **Variational Quantum Classifiers (VQC)** against classical ML models across real-world and scientific datasets.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-quantum-purple)
![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange)

---

## 📖 Overview

This project investigates whether quantum-enhanced machine learning models can compete with — or outperform — their classical counterparts on classification and regression tasks. Using **PennyLane** for quantum circuit simulation, each notebook pairs a Variational Quantum Classifier (VQC) with classical baselines (Logistic Regression, Random Forest, SVM, Neural Networks) on carefully chosen benchmark datasets spanning medical diagnostics, financial fraud, molecular property prediction, and image recognition.

The goal is to provide an honest, reproducible benchmark: same preprocessing, same evaluation metrics, transparent results.

---

## 📂 Repository Structure

```
hybrid-quantum-classical-ml/
│
├── datasets/                        # Raw and preprocessed datasets
│
├── outputs/                         # Plots, confusion matrices, training curves
│
├── results/                         # Aggregated metrics and comparison tables
│
├── breast_cancer_combined.ipynb     # VQC vs Classical on Breast Cancer Wisconsin
├── fraud_detection_benchmark.ipynb  # VQC vs Classical on Credit Card Fraud
├── heart_failure.ipynb              # VQC vs Classical on Heart Failure Clinical Records
├── mnist_quantum_hybrid.ipynb       # Quantum-classical hybrid on MNIST digits
├── qm7.ipynb                        # Quantum regression on QM7 molecular dataset
│
├── requirements.txt                 # Python dependencies
└── LICENSE                          # MIT License
```

---

## 🧪 Experiments & Notebooks

### 1. `breast_cancer_combined.ipynb` — Breast Cancer Classification
Trains a VQC alongside Logistic Regression, SVM, and Random Forest to classify malignant vs benign tumors. PCA is applied to reduce the 30-dimensional feature space to a qubit-compatible size. Evaluates accuracy, F1-score, and ROC-AUC.

### 2. `fraud_detection_benchmark.ipynb` — Credit Card Fraud Detection
Addresses the extreme class imbalance problem in fraud detection (genuine vs fraudulent transactions). Uses SMOTE or undersampling strategies and compares VQC against gradient boosting classifiers under precision-recall tradeoffs.

### 3. `heart_failure.ipynb` — Heart Failure Prediction
Binary classification predicting mortality risk from clinical features (ejection fraction, serum creatinine, etc.). Evaluates whether the quantum model can extract discriminative patterns from a small clinical tabular dataset.

### 4. `mnist_quantum_hybrid.ipynb` — MNIST Quantum-Classical Hybrid
A hybrid pipeline where classical convolutional layers extract features from handwritten digit images, and a quantum circuit acts as the final classifier layer. Demonstrates the hybrid paradigm on a subset of MNIST.

### 5. `qm7.ipynb` — QM7 Molecular Property Prediction
Regression task predicting atomization energies of small organic molecules. The QM7 dataset provides Coulomb matrix representations; this notebook explores whether quantum kernels or VQC layers can capture molecular structure.

---

## 📊 Datasets

All datasets are publicly available. Links and citations are provided below.

| Dataset | Task | Source | License |
|---|---|---|---|
| **Breast Cancer Wisconsin (Diagnostic)** | Binary Classification | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) | CC BY 4.0 |
| **Heart Failure Clinical Records** | Binary Classification | [UCI ML Repository / Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) | CC BY 4.0 |
| **Credit Card Fraud Detection** | Binary Classification | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | DbCL v1.0 |
| **MNIST Handwritten Digits** | Multi-class Classification | [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) | CC BY-SA 3.0 |
| **QM7** | Regression | [quantum-machine.org](http://quantum-machine.org/datasets/) | CC BY-NC 4.0 |

### Dataset Citations

**Breast Cancer Wisconsin:**
> Wolberg, W., Street, W., & Mangasarian, O. (1995). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

**Heart Failure Clinical Records:**
> Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(1), 16. https://doi.org/10.1186/s12911-020-1023-5

**Credit Card Fraud Detection:**
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. In *Symposium on Computational Intelligence and Data Mining (CIDM)*, IEEE. Dataset hosted by the Machine Learning Group, ULB: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**MNIST:**
> LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). *The MNIST Database of Handwritten Digits*. http://yann.lecun.com/exdb/mnist/

**QM7:**
> Blum, L. C., & Reymond, J.-L. (2009). 970 Million Druglike Small Molecules for Virtual Screening in the Chemical Universe Database GDB-13. *J. Am. Chem. Soc.*, 131, 8732. https://doi.org/10.1021/ja902302h
> Montavon, G., et al. (2012). Learning Invariant Representations of Molecules for Atomization Energy Prediction. In *Advances in Neural Information Processing Systems* (NeurIPS). http://quantum-machine.org/datasets/

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
git clone https://github.com/manoj-gb/hybrid-quantum-classical-ml.git
cd hybrid-quantum-classical-ml
pip install -r requirements.txt
```

### Launch Notebooks

```bash
jupyter notebook
```

Open any `.ipynb` file from the repo root and run cells top to bottom.

---

## 🔬 Methods

### Quantum Component
- **Framework:** [PennyLane](https://pennylane.ai/) with `default.qubit` simulator
- **Ansatz:** Parameterized quantum circuits using rotation gates (RX, RY, RZ) and CNOT entanglement layers
- **Data Encoding:** Angle encoding (feature values mapped to qubit rotation angles)
- **Optimization:** Gradient descent via parameter-shift rule or classical autograd

### Classical Baselines
- Logistic Regression
- Support Vector Machine (SVM / SVC)
- Random Forest
- Gradient Boosting (XGBoost / sklearn)
- Multi-layer Perceptron (MLP)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Training time comparison

---

## 📈 Results

Aggregated benchmark results are stored in the `results/` folder. Plots and visualization outputs (confusion matrices, ROC curves, loss curves) are in `outputs/`.

> **Note:** Due to the current limitations of quantum simulators (exponential classical overhead), VQC models operate on dimensionality-reduced feature sets. Classical models are evaluated on both full and reduced feature sets for a fair comparison.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [PennyLane](https://pennylane.ai/) | Quantum circuit simulation & differentiation |
| [scikit-learn](https://scikit-learn.org/) | Classical ML baselines & preprocessing |
| [NumPy](https://numpy.org/) | Numerical computing |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |
| [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) | Visualization |
| [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) | Deep learning baselines & hybrid layers |
| [Jupyter](https://jupyter.org/) | Interactive notebooks |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Dataset licenses vary; please refer to the [Datasets](#-datasets) section above and comply with each dataset's individual license before redistribution or commercial use. Notably:
- The **QM7** dataset is under CC BY-NC 4.0 (non-commercial).
- The **Credit Card Fraud** dataset is under the Open Database License (DbCL v1.0).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or raise an issue.

---

## ✉️ Contact

**Manoj G B** — [GitHub @manoj-gb](https://github.com/manoj-gb)
