# AML Transaction Monitoring - Anomaly Detection

Unsupervised machine learning pipeline to detect suspicious transaction patterns in banking and blockchain data. Built with Python, Scikit-learn, Pandas, TensorFlow, and Flask.

## Overview

End-to-end anomaly detection system for Anti-Money Laundering compliance. Engineers fraud detection features, trains two unsupervised models, and exposes a real-time risk scoring API for fintech and crypto clients.

## Results

| Model | Precision | Recall | F1 Score | False Positive Rate |
|-------|-----------|--------|----------|---------------------|
| Autoencoder | 88.89% | 100.00% | 94.12% | 0.10% |
| Isolation Forest | 81.44% | 98.75% | 89.27% | 0.18% |

## Features

- 50,000 transactions with engineered features including transaction frequency, amount deviation, and time of day patterns
- Two model architectures: Isolation Forest and Autoencoder
- Real-time risk scoring API with Flask
- Interactive web demo with risk calculator and anomaly visualization scatter plot

## Live Demo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12CPIXY0UTaB9DxAKeLr1w60r_qyRWY_g?usp=sharing)

**Live Demo:** [https://aml-monitor-rxzy.onrender.com](https://aml-monitor-rxzy.onrender.com)

## Tech Stack

Python, Scikit-learn, Pandas, NumPy, Matplotlib, TensorFlow, Keras, Flask, Gradio

## Deployment

The model is deployed as a Flask web application on Render with a real-time risk calculator, anomaly detection scatter plot visualization, and model performance metrics dashboard.
