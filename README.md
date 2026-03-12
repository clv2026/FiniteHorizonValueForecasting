# Modular Probabilistic Engagement‑State Modeling for Interpretable Finite‑Horizon Revenue Forecasting


<p align="center">
  <img src=Fig1.png>
</p>

This repository contains the official implementation accompanying the paper: **_Modular Probabilistic Engagement‑State for Interpretable Finite‑Horizon Revenue Forecasting_**

The framework provides a fully modular and extensible pipeline for **engagement‑driven multi‑step forecasting**, separating behavioral evolution (engagement‑state transitions) from downstream value realization.  
It supports horizon‑aware diagnostics, interpretable intermediate representations, and competitive predictive performance compared to strong direct regression baselines.

---

## 🚀 Getting Started

```bash
git clone https://github.com/clv2026/FiniteHorizonValueForecasting
cd FiniteHorizonValueForecasting
```

The folder contains a `requirements.txt` file with all required libraries.

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup (Manual Download Required)

The Online Retail II dataset **must be downloaded manually**

1. Download the dataset manually

    Download the ZIP file from the official UCI Machine Learning Repository:

    🔗 https://archive.ics.uci.edu/dataset/502/online+retail+ii

2. Extract the dataset: 

    Unzip the downloaded archive. Inside, you will find the Excel file, typically named: ```online_retail_II.xlsx```

3. Move the extracted file into: ```classes/data/```


## 📁 Repository Structure
```
FiniteHorizonValueForecasting/
│
├── classes/
│   ├── data/
│   │   └── online_retail_II.xlsx
│   ├── config.yaml               # Configuration file
│   ├── main.py                   # End-to-end pipeline: preprocessing → state modeling → value forecasting
│   ├── preprocessing.py          # Data loading, cleaning, temporal aggregation (weekly/monthly)
│   ├── engage2value.py           # Engagement-state models, transition models, SAM/SCR value mapping
│   └── ablation.py               # Reproduce ablation studies from the paper
│
├── example.ipynb                 # Walkthrough notebook demonstrating the full modular workflow
└── requirements.txt          
```
---

## ⚙️ Configuration

Both `main.py` and `ablation.py` read their runtime parameters from a single configuration file: `config.yaml`

This file controls preprocessing settings, forecasting horizon, and optional benchmarking or test‑mode behavior.

### **Example `config.yaml`**
```yaml
freq: 'M'                           # Temporal aggregation frequency:
                                    # 'M'  = 1 month
                                    # 'W'  = 1 week
period: 2                           # Forecast horizon of the selected freq
test_mode: false                    # If true, uses a reduced dataset for faster runs
data_path: "classes/data"           # Path to dataset folder
data_file: "online_retail_II.xlsx"  # Input data file
save_csv: false                     # Export fully processed dataset as CSV (optional)
benchmarks: false                   # Run benchmark models (Naive Persistence, Mode, etc.)
```

---

## ✨ Reproducing Paper Results

To explore the workflow interactively, open:
```bash
example.ipynb
```

The full experimental pipeline (state prediction, multi-step calibration, and revenue forecasting) can be reproduced using:
```bash
python classes/main.py
```

Ablation experiments:
```bash
python classes/ablation.py
```

---

## Overview

This library enables researchers and practitioners to:

- Convert event‑level data into temporal snapshots (weekly or monthly)
- Infer interpretable engagement states from behavioral features
- Model short‑term engagement‑state transitions using:
  - **M‑ETM** — empirical Markov transitions  
  - **GB‑ETM** — feature‑conditioned probabilistic transitions
- Propagate multi-step state distributions to arbitrary forecast horizons
- Translate engagement states into monetary predictions via:
  - **State‑Average Mapping (SAM)**
  - **State‑Conditional Regression (SCR)**
- Run ablation studies and reproduce experiments from the paper
- Explore the full pipeline via an example notebook

The modular design mirrors the paper’s methodology, exposing each component for analysis, diagnostics, and flexible substitution.

---

## 🧩 Module Descriptions

`example.ipynb`, walkthrough notebook demonstrating the full modular workflow:
- Data preparation  
- Engagement-state generation  
- Transition modeling  
- Multi-step propagation  
- Reconstructing expected value  
- Horizon-aware diagnostics  

This notebook is the best entry point for new users.

`classes/main.py`, runs the full modular workflow:
- Data ingestion  
- Temporal snapshot construction  
- Engagement‑state assignment  
- Transition modeling (M‑ETM or GB‑ETM)  
- Multi‑horizon propagation  
- State-to-value reconstruction (SAM or SCR)  
- Evaluation metrics  

`classes/preprocessing.py`, implements:
- Raw data loading  
- Cleaning and filtering of event streams  
- Aggregation into weekly or monthly snapshots  
- Feature construction (recency, frequency, intensity, rolling windows)  
- State labeling utilities  

`classes/engage2value.py`, contains the core forecasting components:
- Engagement-state definitions
- M‑ETM (Markov transitions)
- GB‑ETM (feature‑conditioned transitions)
- Multi-step state propagation
- SAM (state-average mapping)
- SCR (state-conditional regressors)
- Evaluation utilities and benchmarks

`classes/ablation.py`, runs the three ablation setups described in the paper, allowing users to replicate:
- Reduced feature sets
- Randomized transition models
- Direct value‑only regressors

## 📄 Dataset Acknowledgment

This project uses the **Online Retail II** dataset from the UCI Machine Learning Repository:
> Chen, D. (2012). Online Retail II [Dataset].
> UCI Machine Learning Repository.
> https://doi.org/10.24432/C5CG6D.
