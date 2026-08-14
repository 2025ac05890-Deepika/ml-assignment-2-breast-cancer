
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Breast Cancer Classification")

st.write(
    """
    This application evaluates machine learning classification
    models on the Breast Cancer Wisconsin (Diagnostic) dataset.
    """
)

st.info(
    "Class 0 = Benign | Class 1 = Malignant"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression":
            joblib.load("model/logistic_regression.pkl"),

        "Decision Tree":
            joblib.load("model/decision_tree.pkl"),

        "kNN":
            joblib.load("model/knn.pkl"),

        "Naive Bayes":
            joblib.load("model/naive_bayes.pkl"),

        "Random Forest":
            joblib.load("model/random_forest.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler


models, scaler = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys())
)


# ============================================================
# DATA UPLOAD
# ============================================================

st.header("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is None:

    st.warning(
        "Please upload the test_data.csv file."
    )

    st.stop()


# ============================================================
# READ DATA
# ============================================================

test_data = pd.read_csv(uploaded_file)

st.success("Test data uploaded successfully!")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Number of Samples",
        test_data.shape[0]
    )

with col2:
    st.metric(
        "Number of Columns",
        test_data.shape[1]
    )


with st.expander("View Test Data"):

    st.dataframe(
        test_data,
        use_container_width=True
    )


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

target_column = "diagnosis"

if target_column not in test_data.columns:

    st.error(
        "The uploaded file must contain a 'diagnosis' column."
    )

    st.stop()


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_test = test_data.drop(
    columns=[target_column]
)

y_test = test_data[target_column]


# ============================================================
# PREPROCESSING
# ============================================================

if selected_model in [
    "Logistic Regression",
    "kNN",
]:

    X_input = scaler.transform(X_test)

else:

    X_input = X_test


# ============================================================
# MODEL PREDICTION
# ============================================================

model = models[selected_model]

predictions = model.predict(X_input)

probabilities = model.predict_proba(X_input)


# ============================================================
# EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    probabilities[:, 1]
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    predictions
)


# ============================================================
# DISPLAY METRICS
# ============================================================

st.header("2. Evaluation Metrics")

st.subheader(selected_model)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with col2:
    st.metric(
        "AUC",
        f"{auc:.4f}"
    )

with col3:
    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with col5:
    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

with col6:
    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("3. Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predictions
)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Benign",
        "Malignant"
    ],
    yticklabels=[
        "Benign",
        "Malignant"
    ],
    ax=ax
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(
    selected_model + " - Confusion Matrix"
)

st.pyplot(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("4. Classification Report")

report = classification_report(
    y_test,
    predictions,
    target_names=[
        "Benign",
        "Malignant"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# PREDICTIONS
# ============================================================

st.header("5. Predictions")

prediction_data = test_data.copy()

prediction_data["Predicted"] = predictions

prediction_data["Actual Label"] = (
    prediction_data["diagnosis"]
    .map({
        0: "Benign",
        1: "Malignant"
    })
)

prediction_data["Predicted Label"] = (
    prediction_data["Predicted"]
    .map({
        0: "Benign",
        1: "Malignant"
    })
)

st.dataframe(
    prediction_data,
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "M.Tech AIML/DSE | Machine Learning Assignment 2"
)
