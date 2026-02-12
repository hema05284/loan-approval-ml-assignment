# Loan Approval Prediction 
## a. Problem Statement

The objective of this project is to design and evaluate multiple machine learning classification models to predict whether a loan application will be approved or rejected. The focus of the study is to compare the performance of different classifiers using standard evaluation metrics and demonstrate the results through an interactive Streamlit web application.

---

## b. Dataset Description

The Loan Approval dataset contains demographic and financial information of loan applicants. The features include applicant income, co-applicant income, loan amount, credit history, education level, employment status, marital status, dependents, and property area.

- **Number of Instances:** More than 600  
- **Number of Features:** More than 12  
- **Problem Type:** Binary Classification  

### Target Variable:
- `Loan_Status`
  - `Y` → Loan Approved  
  - `N` → Loan Not Approved  

The dataset was obtained from a publicly available repository and was preprocessed to handle missing values, encode categorical features, and normalize numerical attributes before training the models.

---

## c. Models Used and Evaluation Metrics

The following classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using the following performance metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

# Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.7886 | 0.7520 | 0.7596 | 0.9875 | 0.8587 | 0.5358 |
| Decision Tree | 0.7480 | 0.6388 | 0.7333 | 0.9625 | 0.8324 | 0.4200 |
| kNN | 0.7642 | 0.5000 | 0.7383 | 0.9875 | 0.8449 | 0.4768 |
| Naive Bayes | 0.7805 | 0.7265 | 0.7573 | 0.9750 | 0.8525 | 0.5086 |
| Random Forest (Ensemble) | 0.7805 | 0.7892 | 0.7573 | 0.9750 | 0.8525 | 0.5086 |
| XGBoost (Ensemble) | 0.7561 | 0.7311 | 0.7604 | 0.9125 | 0.8295 | 0.4350 |

*Metric values obtained from model evaluation performed on BITS Virtual Lab.*

---

# Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieved high recall (0.9875) and strong F1 score, indicating effective identification of approved loan applications while maintaining balanced precision. |
| Decision Tree | Demonstrated good recall but lower AUC and MCC compared to other models, suggesting possible overfitting and weaker probability ranking ability. |
| kNN | Very high recall but AUC of 0.5000 indicates limited ability to distinguish between classes probabilistically. Performance is sensitive to feature scaling. |
| Naive Bayes | Provided stable and consistent performance with high recall and competitive MCC, showing strong baseline classification capability. |
| Random Forest (Ensemble) | Achieved the highest AUC (0.7892), indicating strong discrimination ability and improved generalization through ensemble learning. |
| XGBoost (Ensemble) | Delivered balanced precision and recall but slightly lower overall performance compared to Random Forest on this dataset. |

---

## Execution Environment

All models were trained and evaluated on **BITS Virtual Lab**, and a screenshot of the execution output has been included in the final submission PDF as proof.

---

## Streamlit Application

The trained models and evaluation results are demonstrated using a Streamlit web application. The app allows users to:

- Upload test datasets (CSV format)  
- Select a classification model  
- View evaluation metrics  
- Visualize confusion matrix  

The application is deployed using Streamlit Community Cloud.

---

## Repository Structure

