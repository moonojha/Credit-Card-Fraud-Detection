# Credit Card Fraud Detection — MLOps Pipeline

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions, covering model comparison, hyperparameter tuning, experiment tracking, and deployment via a containerized REST API.

## Overview

This project tackles a highly imbalanced binary classification problem (fraud vs. non-fraud, ~0.17% positive class) using the Kaggle Credit Card Fraud dataset. Rather than stopping at a notebook, the project is built as a full pipeline: trained models are tracked with MLflow, tuned with Optuna, and served through a FastAPI application packaged in Docker.

**Chosen model:** XGBoost (compared against Random Forest and LightGBM)

## Results

Evaluated on a held-out test set (56,962 transactions, 98 fraud cases), using a decision threshold of 0.15 (tuned to prioritize recall).

**Precision:** Non-fraud -> 1.000     Fraud -> 0.06

**Recall:** Non-fraud -> 0.98         Fraud -> 0.92

**F1-score:** Non-fraud -> 0.99       Fraud -> 0.12


**Overall accuracy:** 98% (not a meaningful metric on its own given the class imbalance — included for reference only)

**Training time:** 4.84s 

**Prediction time:** 0.32s


**Interpretation:** the model catches 92% of actual fraud cases, at the cost of a high false-positive rate (roughly 1 in 17 flagged transactions is truly fraudulent). This is a deliberate tradeoff — see below.



**Class imbalance:** handled via scale_pos_weight (XGBoost) and class_weight='balanced' (Random Forest) rather than synthetic oversampling (e.g. SMOTE), to avoid introducing synthetic data risk. Combined with threshold tuning based on the precision-recall curve rather than the default 0.5 cutoff.

**Threshold choice (0.15):** in fraud detection, a missed fraud case (false negative) is typically far costlier than a false alarm (false positive), which a human reviewer can quickly dismiss. The threshold was chosen to maximize recall while keeping precision non-trivial, rather than optimizing for accuracy or F1 alone.

**Feature selection:** top 12 features selected by absolute correlation with the target, computed on the training set only to avoid data leakage.

**Feature scope:** Amount and an engineered Hour (derived from Time) were evaluated during feature selection but did not rank in the final top 12 — the deployed model uses only the anonymized PCA-derived features (V1–V28).

**Model comparison:** Random Forest, XGBoost, and LightGBM were all trained and compared; XGBoost was selected as the final model based on precision-recall performance.

**Hyperparameter tuning:** XGBoost was tuned via Optuna.

**Experiment tracking:** all training runs, hyperparameters, and metrics logged via MLflow.


# Running Locally

**1. Clone and set up Kaggle credentials**

The notebook downloads the dataset directly via the Kaggle API. Get an API token at kaggle.com/settings → API → Create New Token, and place kaggle.json in ~/.kaggle/.

**2. Install dependencies**

pip install -r requirements.txt

**3. Run the notebook (optional — a trained model is already included in models/)**

jupyter notebook notebooks/Credit_Card_Fraud_Detection_MLOps.ipynb

**4. Run the API**

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Visit http://localhost:8000/docs for interactive API documentation.

Or, with Docker:

docker build -t fraud-api .
docker run -p 8000:8000 fraud-api


**Example request:**

jsonPOST /predict

{

  "features": [-0.544942347836109, 0.378142770624644, -0.923745559120948, 0.634169644382178, 0.956241175217331, 0.924552664874382, -0.41159257488638, -0.907725430357759, 2.4035957814499, -0.224747261672232, 1.07234945559604, -0.184886542413332]

}

**Example response:**

json{

  "prob_fraud": 0.87,

  "is_fraud": 1
  
}


**Note:** because the dataset's features are PCA-anonymized for privacy (a realistic constraint in real-world fraud systems), the API doesn't accept human-interpretable input. The /sample_transaction endpoint returns a real transaction from the test set that can be passed directly to /predict for demonstration purposes.



# Limitations & Next Steps


**Precision-recall tradeoff:** the current threshold heavily favors recall. In a production setting, this threshold would be tuned in collaboration with whoever handles fraud review, based on their actual review capacity.

**No drift monitoring:** a production system would need to track feature distribution drift over time; not implemented here.

**Static threshold:** currently hardcoded (0.15); a production system might adjust it dynamically based on transaction context (e.g. amount, merchant category).

**No live/public deployment:** currently runs locally or via Docker; a next step would be deploying to a cloud service (e.g. Render, Railway, Fly.io) for a persistent public demo.


## Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), provided by the Machine Learning Group at ULB. Not included in this repo due to size and redistribution terms; download via the Kaggle API as described above.