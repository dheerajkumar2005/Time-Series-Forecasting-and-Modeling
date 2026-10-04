# Time Series Forecasting & Non-Parametric Anomaly Detection

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Statsmodels](https://img.shields.io/badge/Modeling-Statsmodels%20%7C%20ARIMA%20%7C%20SARIMA-green.svg)]()
[![Anomaly Detection](https://img.shields.io/badge/Detection-KDE%20%7C%20Epanechnikov%20Kernel-orange.svg)]()
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![IIT Bombay](https://img.shields.io/badge/IIT%20Bombay-CS215%20Data%20Analysis-red.svg)](https://www.cse.iitb.ac.in/)

> **Academic Affiliation**: Course Project for **CS 215: Data Analysis and Interpretation**, IIT Bombay  
> **Guide**: **Prof. Sunita Sarawagi**  
> **Author**: **Dheeraj Kumar Maradana** ([@dheerajkumar2005](https://github.com/dheerajkumar2005))

---

## 📌 Executive Summary

This repository implements statistical time-series forecasting pipelines alongside non-parametric anomaly and fraud detection models.

The project is structured into two analytical components:
1. **Statistical Time Series Modeling**: Stationarity diagnosis via the Augmented Dickey-Fuller (ADF) test, order selection through Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions, and forecast modeling leveraging **ARIMA, SARIMA**, and **Exponential Smoothing (ETS)** models.
2. **Non-Parametric Anomaly & Outlier Detection**: Implementation of multi-dimensional **Kernel Density Estimation (KDE)** using the optimal **Epanechnikov Kernel** to identify rare, low-probability events and anomalous transactions from rolling window feature dynamics.

---


## 🔬 Core Implementations

### 1. Stationarity & Order Selection (`Q1_c_lin.ipynb`)
* **Unit Root Testing**: Evaluates null hypothesis for non-stationarity using ADF statistics and $p$-values.
* **Autocorrelation Analysis**: Uses sample ACF/PACF plots to bound initial AR ($p$) and MA ($q$) orders.
* **Seasonal Modeling**: Integrates seasonal differencing ($D$) and seasonal lag components ($P, Q$) over cyclical periods.
* **Evaluation Metrics**: Compares AIC, BIC, RMSE, and MAPE across competing parameterizations.

### 2. Epanechnikov Kernel Anomaly Detection (`Q2_c_rolling_Stats.ipynb`)
* **Epanechnikov Density Formulation**: Employs the bounded-support, minimum-mean-integrated-squared-error Epanechnikov kernel:
  $$K(u) = \frac{3}{4}(1 - u^2) \quad \text{for } |u| \le 1$$
* **Rolling Statistics Engine**: Computes rolling means, variances, and skewness across multi-dimensional feature spaces.
* **Low-Probability Scoring**: Ranks events by log-likelihood density $\ln \hat{f}(x)$, isolating anomalous outliers deviating significantly from nominal distributions.

---

## 📁 Repository Structure

```
├── Q1_c_lin.ipynb              # Complete ARIMA / SARIMA / ETS time series analysis notebook
├── Q2_c_rolling_Stats.ipynb    # Rolling stats feature extraction & KDE anomaly detection notebook
├── Q1/ & Q2/                   # Intermediate scripts, figures, and benchmark logs
└── a4.pdf                      # Comprehensive project specification and problem statements
```

---

## 🚀 Getting Started

### Prerequisites & Dependencies
```bash
git clone https://github.com/dheerajkumar2005/Time-Series-Forecasting-and-Modeling.git
cd Time-Series-Forecasting-and-Modeling

pip install numpy pandas scipy statsmodels matplotlib seaborn jupyter scikit-learn
```

### Running the Notebooks
```bash
jupyter notebook Q1_c_lin.ipynb
jupyter notebook Q2_c_rolling_Stats.ipynb
```
