# AML Transaction Monitoring - Anomaly Detection

Unsupervised machine learning pipeline to detect suspicious transaction patterns in simulated banking and blockchain data. Built with Python, Scikit-learn, Pandas, TensorFlow, and Flask.

## Overview

End-to-end anomaly detection system for Anti-Money Laundering compliance. Generates synthetic transaction data, engineers fraud detection features, trains two unsupervised models, and exposes a real-time risk scoring API for fintech and crypto clients.

## Results

| Model | Precision | Recall | F1 Score | False Positive Rate |
|-------|-----------|--------|----------|---------------------|
| Autoencoder | 88.89% | 100.00% | 94.12% | 0.10% |
| Isolation Forest | 81.44% | 98.75% | 89.27% | 0.18% |

## Features

- Synthetic transaction data generation with 50,000 records and 0.8% fraud rate
- Feature engineering: Amount, Hour of Day, Sender Frequency, Amount Deviation
- Two model architectures: Isolation Forest and Autoencoder
- Real-time risk scoring API with Flask and Gradio
- Interactive web demo with risk calculator and anomaly visualization

## Live Demo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12CPIXY0UTaB9DxAKeLr1w60r_qyRWY_g?usp=sharing)

**Live Demo:** [https://aml-monitor-rxzy.onrender.com](https://aml-monitor-rxzy.onrender.com)

## Quick Start

1. Open the notebook in Google Colab and run all cells
2. No external data required - everything is generated in the notebook
3. Or visit the live demo to test the risk calculator instantly

## Tech Stack

Python, Scikit-learn, Pandas, NumPy, Matplotlib, TensorFlow, Keras, Flask, Gradio

## Deployment

The model is deployed as a Flask web application on Render with:
- Real-time risk calculator with color-coded results
- Anomaly detection scatter plot visualization
- Model performance metrics dashboard
