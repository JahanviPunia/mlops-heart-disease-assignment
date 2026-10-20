# Heart Disease Classification — MLOps Assignment

## 1. Project Overview

This project explores the machine learning lifecycle using the UCI Heart Disease dataset. I used Google Colab to explore the data, prepare features, compare two classifiers, track experiments, save a trained model, and test a FastAPI prediction service.

## 2. Objectives

- Explore the dataset and identify missing values.
- Compare Logistic Regression and Random Forest.
- Evaluate performance using five-fold stratified cross-validation.
- Track experiments with MLflow.
- Serve predictions through FastAPI.
- Add automated tests and continuous integration.
- Prepare Docker configuration for reproducible deployment.

## 3. Dataset

The project uses the UCI Heart Disease dataset accessed through the `ucimlrepo` package.

- Records: 303
- Input features: 13
- Task: Binary classification
- Target 0: No heart disease indicated in the dataset
- Target 1: A non-zero original target value, indicating heart disease in this dataset

The original target was converted into a binary target for this experiment. Missing values in `ca` and `thal` are handled during preprocessing.

## 4. Exploratory Data Analysis

The initial analysis covered dataset dimensions, target distribution, selected feature distributions, feature correlations, and missing values. The cleaned dataset and class-balance chart are included in the project.

## 5. Model Comparison

I compared two classifiers using five-fold stratified cross-validation.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.835 | 0.832 | 0.806 | 0.910 |
| Random Forest | 0.842 | 0.860 | 0.784 | 0.908 |

Logistic Regression was selected based on mean ROC-AUC. Random Forest achieved slightly higher accuracy and precision, while Logistic Regression achieved higher recall and ROC-AUC.

## 6. Experiment Tracking

MLflow records model experiments and evaluation metrics. The training script saves the model comparison, training summary, and selected model artifact.

## 7. API

The FastAPI application provides:

- `/health` — checks service and model availability.
- `/predict` — returns a binary prediction and positive-class probability.
- `/metrics` — exposes a basic API request counter.

Request logging and input validation are included.

## 8. Testing

Four API tests passed in Google Colab:

- Health endpoint
- Valid prediction
- Rejection of unexpected input
- Metrics endpoint

The GitHub Actions workflow has been created but still needs to be verified on GitHub.

## 9. Technology Stack

Python, pandas, NumPy, scikit-learn, MLflow, FastAPI, pytest, GitHub Actions, Docker, and Kubernetes configuration.

## 10. Current Status

Data exploration, model comparison, experiment tracking, model saving, and API smoke tests have been completed in Google Colab. Automated CI and container deployment remain to be verified. Cloud deployment will be marked complete only after a successful deployment and endpoint test.

## 11. Limitations and Disclaimer

This is an educational assignment using a relatively small dataset. Cross-validation results do not guarantee performance on other populations. The model is not clinically validated and must not be used for medical diagnosis or treatment decisions.
