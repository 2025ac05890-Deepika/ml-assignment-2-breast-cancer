# AIMLCZC565_MLassignment2

Machine Learning Assignment - 2 using model comparison and displays the results in Streamlit application

# Breast Cancer Classification

## a. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models to predict whether a breast tumor is Benign or Malignant. The target variable is a binary classification problem where the tumor is classified as either Benign (0) or Malignant (1). The goal is to evaluate different machine learning approaches and identify the model that provides the strongest overall predictive performance.

## b. Dataset Description

The predictive models are trained and evaluated on the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the **UCI Machine Learning Repository**, accessed through the scikit-learn dataset collection.

The dataset contains numerical measurements computed from digitized images of breast tissue samples. The features describe characteristics of cell nuclei, including measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

- **Problem Type:** Binary Classification
- **Number of Instances:** 569
- **Number of Features:** 30
- **Number of Classes:** 2
- **Target Variable:** diagnosis
- **Classes:** Benign (0) and Malignant (1)

The dataset satisfies the assignment requirements of a minimum of 500 instances and a minimum of 12 features.

The data was divided into training and testing sets. The test dataset contains 114 samples and 30 input features.

## c. GitHub Repository Link

https://github.com/2025ac05890-Deepika/ml-assignment-2-breast-cancer/tree/main

## d. Models Used & Evaluation Metrics

The dataset was evaluated using five different classification algorithms. The models were trained on the training data and evaluated on the held-out test dataset.

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **K-Nearest Neighbor (KNN)** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Gaussian Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Random Forest (Ensemble)** | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

## e. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest Accuracy of **0.9825** and the highest MCC of **0.9623**. It also achieved an excellent AUC of **0.9954**, with Precision, Recall, and F1 Score all equal to **0.9861**, demonstrating strong and balanced classification performance. |
| **Decision Tree** | Achieved an Accuracy of **0.9123** and an AUC of **0.9157**. It had the lowest Accuracy, AUC, and MCC among the five models. Its Recall of **0.9028** and F1 Score of **0.9286** indicate reasonable classification performance, although it performed below the other models overall. |
| **KNN** | Delivered strong performance with an Accuracy of **0.9561**, Precision of **0.9589**, Recall of **0.9722**, and F1 Score of **0.9655**. Its MCC of **0.9054** also indicates strong classification performance. |
| **Gaussian Naive Bayes** | Achieved an Accuracy of **0.9386** and an AUC of **0.9878**. It provided good class discrimination, with Precision of **0.9452**, Recall of **0.9583**, F1 Score of **0.9517**, and MCC of **0.8676**. |
| **Random Forest (Ensemble)** | Achieved an Accuracy of **0.9561** and a high AUC of **0.9937**. Its Recall of **0.9722** and F1 Score of **0.9655** demonstrate strong classification performance. |
| **Overall Winner** | **Logistic Regression** is selected as the overall winner based on the current test results. It achieved the highest Accuracy (**0.9825**), Precision (**0.9861**), Recall (**0.9861**), F1 Score (**0.9861**), and MCC (**0.9623**). It also achieved a very high AUC of **0.9954**. |


