# Heart Disease Classification — MLOps Assignment

## 1. Project Overview

This project focuses on building and evaluating a machine learning model for heart disease classification using the UCI Heart Disease dataset. The objective is to understand the complete machine learning lifecycle, from data exploration and model development to experiment tracking, API development, testing, and deployment preparation.

I used Google Colab to explore the dataset, prepare the features, compare two classification algorithms, and save the selected model for inference.

## 2. Project Objectives

- Explore the dataset and understand its features and target distribution.
- Handle missing values and prepare data for model training.
- Compare Logistic Regression and Random Forest classifiers.
- Evaluate performance using stratified cross-validation.
- Track experiments and model results using MLflow.
- Develop a FastAPI service for generating predictions.
- Prepare automated testing and deployment configurations.
- Document the workflow to support reproducibility.

## 3. Dataset

The project uses the UCI Heart Disease dataset, accessed through the `ucimlrepo` Python package.

- Dataset size: 303 records
- Number of original input features: 13
- Classification task: Binary classification
- Target class `0`: No heart disease indicated in the dataset
- Target class `1`: Heart disease indicated by a non-zero original target value

The original target contains multiple severity values. For this project, I converted it into a binary target to simplify the classification task.

## 4. Exploratory Data Analysis

The initial analysis included:

- Checking the dataset dimensions and column types.
- Examining the distribution of the target classes.
- Visualizing selected numerical features.
- Reviewing correlations between features.
- Identifying missing values.
- Preparing a cleaned dataset for model development.

The dataset contains missing values in the `ca` and `thal` features, which are handled during preprocessing.

## 5. Model Development and Evaluation

I compared two classification algorithms using five-fold stratified cross-validation:

1. Logistic Regression
2. Random Forest Classifier

The preprocessing and classification steps were combined into scikit-learn pipelines to keep the workflow consistent.

### Initial Cross-Validation Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.835 | 0.832 | 0.806 | 0.910 |
| Random Forest | 0.842 | 0.860 | 0.784 | 0.908 |

### Model Selection

Logistic Regression was selected because it achieved the higher mean ROC-AUC of approximately 0.910. Random Forest achieved slightly higher accuracy and precision, while Logistic Regression had higher recall and ROC-AUC.

The final selection was based on the chosen evaluation metric rather than accuracy alone.

## 6. Experiment Tracking with MLflow

MLflow was used to record model experiments and evaluation metrics.

The experiment tracking workflow includes:

- Recording model names and experiment parameters.
- Logging cross-validation metrics.
- Comparing model performance.
- Recording the selected model.
- Saving the trained model artifact.

This helps maintain a record of the experiments performed during model development.

## 7. API Development

I created a FastAPI service to expose the trained model through HTTP endpoints.

The API includes:

- `/health` — checks service and model availability.
- `/predict` — accepts patient feature values and returns a prediction and positive-class probability.
- `/metrics` — exposes a basic API request counter.

The prediction response includes the model name and an educational-use disclaimer.

## 8. Testing

Initial API smoke tests were executed in Google Colab using FastAPI's `TestClient`.

The tests checked:

- Whether the health endpoint returned a successful response.
- Whether the prediction endpoint accepted valid input.
- Whether the response included the prediction and probability.
- Whether the metrics endpoint returned the request counter.

Further automated tests will be incorporated into the repository's CI workflow.

## 9. Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- MLflow
- FastAPI
- GitHub and GitHub Actions
- Docker
- Kubernetes configuration

## 10. Current Status and Next Steps

The initial data exploration, model comparison, MLflow experiment tracking, model artifact creation, and API smoke tests have been completed in Google Colab.

The next steps are to organize the source code in GitHub, add reproducibility instructions, validate automated tests, prepare the Docker image, and verify the deployment workflow.

Deployment will be marked complete only after the service has been built, deployed, and tested successfully.

## 11. Limitations

- The dataset is relatively small and may not represent broader patient populations.
- Cross-validation metrics are estimates of model performance and do not guarantee generalization to new populations.
- The current API is an educational demonstration, not a clinically validated service.
- Predictions must not be used for medical diagnosis or treatment decisions.

## 12. Disclaimer

This project was developed for educational purposes as part of an MLOps assignment. The model is not intended for clinical use, and its predictions should not be interpreted as medical advice.
