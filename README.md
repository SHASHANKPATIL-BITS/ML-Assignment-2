# Machine Learning Classification: Heart Disease Prediction

### a. Problem Statement
The objective of this project is to build and evaluate multiple machine learning classification models to accurately predict the presence of heart disease in patients based on their medical attributes. The models are deployed via a Streamlit web application to allow interactive predictions on test data.

### b. Dataset Description
- **Source:** Kaggle (Heart Disease Dataset)
- **Minimum Instance Size:** 1025 rows
- **Minimum Feature Size:** 13 features + 1 target column (14 total)
- **Description:** The dataset contains clinical features such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, and maximum heart rate achieved. The target variable is binary, indicating the presence (1) or absence (0) of heart disease.

### c. Github Repository Link
(https://github.com/SHASHANKPATIL-BITS/ML-Assignment-2/tree/main)

### d. Models Used and Comparison Table
The following five classification models were trained and evaluated on the dataset: Logistic Regression, Decision Tree, k-Nearest Neighbors (kNN), Naive Bayes, and Random Forest. 

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7951 | 0.8787 | 0.7563 | 0.8738 | 0.8108 | 0.5973 |
| Decision Tree | 0.9854 | 0.9854 | 1.0000 | 0.9709 | 0.9852 | 0.9712 |
| kNN | 0.8341 | 0.9486 | 0.8000 | 0.8932 | 0.8440 | 0.6727 |
| Naive Bayes | 0.8000 | 0.8706 | 0.7541 | 0.8932 | 0.8178 | 0.6102 |
| Random Forest (Ensemble)| 0.9854 | 1.0000 | 1.0000 | 0.9709 | 0.9852 | 0.9712 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed adequately as a baseline with ~79.5% accuracy, but struggled with precision (75.6%). This suggests the dataset has non-linear relationships that a simple linear decision boundary cannot fully capture. |
| **Decision Tree** | Outstanding performance with ~98.5% accuracy and perfect precision (1.0). It adapted extremely well to the non-linear features of the dataset. |
| **kNN** | Solid middle-ground performer (~83.4% accuracy). It benefited heavily from the scaling applied during preprocessing but was ultimately outperformed by tree-based algorithms. |
| **Naive Bayes** | Showed the lowest overall accuracy (80.0%) and MCC (0.61), likely because its core assumption of feature independence does not hold perfectly true for complex medical data. |
| **Random Forest (Ensemble)** | Achieved ~98.5% accuracy, perfect precision (1.0), and a flawless AUC of 1.0. Maintained the high accuracy of the Decision Tree while ensuring better generalization. |
| **Overall Winner for your dataset?** | **Random Forest** |
