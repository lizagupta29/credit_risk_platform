# AI-Powered Credit Risk Intelligence Platform

## Overview

This project is an AI-powered Credit Risk Intelligence Platform developed using machine learning, explainable AI, and large language models.

The platform helps financial institutions:

* Predict loan default risk
* Generate risk scores
* Explain model decisions
* Derive business-readable rules
* Query data using natural language
* Explore credit data through an interactive dashboard

---

## Features

### Exploratory Data Analysis

* Missing value analysis
* Demographic insights
* Financial insights
* Default distribution
* Correlation analysis

### Machine Learning

* LightGBM Classifier
* Class imbalance handling
* Risk score generation
* Risk band classification

### Explainable AI

* SHAP Summary Plot
* SHAP Waterfall Plot
* Feature Importance

### Talk-to-Data Chatbot

* Gemini API Integration
* Natural Language to SQL
* Business Insight Generation

### Business Rules

* Human-readable decision rules
* Credit policy support

### Deployment

* Streamlit UI
* Dockerized deployment

---

## Project Structure

(Insert project tree here)

---

## Installation

```bash
git clone <repo-url>
cd credit_risk_platform
pip install -r requirements.txt
```

Create environment file:

```bash
cp .env.example .env
```

Add Gemini API Key.

---

## Create Database

```bash
python create_database.py
```

---

## Train Model

```bash
python src/ml/train.py
```

---

## Evaluate Model

```bash
python src/ml/evaluate.py
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Docker

```bash
docker-compose up --build
```

Application URL:

http://localhost:8501

---

## Future Improvements

* Real-time credit scoring API
* RAG-powered financial knowledge base
* Advanced rule extraction
* Multi-model ensemble scoring
