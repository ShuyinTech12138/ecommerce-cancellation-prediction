# E-Commerce Order Cancellation Prediction

## Overview
Predicting order cancellation risk on 100K+ Brazilian e-commerce transactions using XGBoost with hyperparameter tuning.

## Dataset
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100K orders from 2016–2018.

## Key Findings
- Payment installments and payment type are the strongest predictors of cancellation
- Estimated delivery window and freight value also significantly impact cancellation risk

## Tech Stack
- Python (pandas, NumPy, scikit-learn, XGBoost)
- GridSearchCV for hyperparameter tuning
- SMOTE for class imbalance handling

## How to Run
1. Download dataset from Kaggle link above
2. Place CSV files in the same folder as the script
3. Run: `python ecommerce_cancellation.py`
