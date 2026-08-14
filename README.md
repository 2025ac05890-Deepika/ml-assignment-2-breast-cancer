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

https://github.com/2025ac05890-Deepika/ml-assignment-2-breast-cancer

## d. Models Used & Evaluation Metrics

The dataset was evaluated using five different classification algorithms. The models were trained on the training data and evaluated on the held-out test dataset.

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| **Decision Tree** | 0.9035 | 0.9716 | 0.9189 | 0.8095 | 0.8608 | 0.7911 |
| **K-Nearest Neighbor (KNN)** | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| **Gaussian Naive Bayes** | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| **Random Forest (Ensemble)** | 0.9649 | 0.9944 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

## e. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved very strong performance with an Accuracy of 0.9649 and the highest AUC of 0.9960. Its Precision of 0.9750, Recall of 0.9286, F1 Score of 0.9512, and MCC of 0.9245 indicate strong and balanced classification performance. |
| **Decision Tree** | Achieved an Accuracy of 0.9035 and an AUC of 0.9716. It had the lowest Accuracy, F1 Score, and MCC among the five models. Its Recall of 0.8095 was also lower than the other models. |
| **KNN** | Delivered strong performance with an Accuracy of 0.9561, Precision of 0.9744, Recall of 0.9048, and F1 Score of 0.9383. Its MCC of 0.9058 indicates strong classification performance. |
| **Gaussian Naive Bayes** | Achieved an Accuracy of 0.9211 and an AUC of 0.9891. It provided good class discrimination, although its Accuracy, Recall, F1 Score, and MCC were lower than those of Logistic Regression, KNN, and Random Forest. |
| **Random Forest (Ensemble)** | Achieved an Accuracy of 0.9649 and the highest Precision of 1.0000. It also achieved the highest MCC of 0.9258 and an F1 Score of 0.9500, demonstrating excellent overall classification performance. |
| **Overall Winner** | **Random Forest** is selected as the overall winner based on the combined evaluation metrics. It achieved the same highest Accuracy as Logistic Regression (0.9649), perfect Precision (1.0000), the highest MCC (0.9258), and an F1 Score of 0.9500. Logistic Regression achieved a slightly higher AUC (0.9960), but Random Forest provided the strongest overall balance of the reported metrics. |

## f. Training vs Testing Accuracy

| ML Model Name | Training Accuracy | Testing Accuracy | Difference |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.9868 | 0.9649 | 0.0219 |
| **Decision Tree** | 0.9626 | 0.9035 | 0.0591 |
| **KNN** | 0.9780 | 0.9561 | 0.0219 |
| **Gaussian Naive Bayes** | 0.9451 | 0.9211 | 0.0240 |
| **Random Forest** | 0.9824 | 0.9649 | 0.0175 |

The training and testing accuracies are relatively close for most models, indicating good generalization. The Decision Tree has the largest training-testing difference, while Random Forest has the smallest difference.

## g. Streamlit Application

An interactive Streamlit application was developed to demonstrate the trained classification models.

The application includes model selection, test-data input/upload, predictions, Accuracy, AUC, Precision, Recall, F1 Score, MCC, confusion matrix, and classification report.

## h. Project Structure

```text
ml-assignment-2-breast-cancer/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── logistic_regression.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```

## i. Requirements

The project uses Python, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib, and Streamlit. The required packages are listed in `requirements.txt`.

## j. How to Run the Streamlit Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## k. Streamlit Community Cloud

**Live Streamlit App Link:** To be added after deployment.
