# 🏦 Bank Customer Scoring - BCP Digital Analytics

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-yellowgreen.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)
![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey.svg)

## 📌 Overview

This repository contains the core Machine Learning pipeline and the inference API developed during my final-year engineering internship at **Banque Centrale Populaire (BCP)**. The system predicts which MDM (**Marocains du Monde** / Moroccan Diaspora) customers are most likely to adopt digital banking channels.

> **Production Context:** During the internship, the final architecture was deployed on a distributed 3-node Big Data cluster utilizing **Apache Spark** and **Hadoop (HDFS)** for batch processing. 
> *This specific repository serves as a portfolio demonstration. It focuses on the exploratory Data Science workflow and provides a standalone, containerized version of the real-time inference API.*

## 🧠 Machine Learning Pipeline (CRISP-DM)

The project heavily focuses on data preparation and handling extreme class imbalance:
1. **Exploratory Data Analysis (EDA):** Cleaning multi-source banking data and feature selection.
2. **Data Balancing:** Extensive testing of sampling techniques including SMOTE, Borderline-SMOTE, and CTGAN to solve majority class bias.
3. **Modeling & Benchmarking:** Training and cross-validation of Random Forest, XGBoost, and LightGBM. The final model (Random Forest + Borderline-SMOTE) achieved an F1-Score of 0.88.

## 📂 Repository Structure

```text
Bank-Customer-Scoring-App/
├── notebooks_and_eda/       # Data Science research and modeling
│   ├── analyse_mdm.ipynb    # Feature engineering & EDA
│   ├── metabalance.ipynb    # Class imbalance handling
│   └── baseline_code.ipynb  # Model benchmarking & metrics
├── app.py                   # Flask REST API for real-time inference
├── Dockerfile               # Containerization instructions
└── requirements.txt         # Python dependencies