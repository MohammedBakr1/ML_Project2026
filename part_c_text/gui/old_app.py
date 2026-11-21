
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)

from preprocessing import make_cleaner



BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

ARTIFACTS_PATH = DATASET_DIR / "artifacts.joblib"
NUMPY_PARAMS_PATH = DATASET_DIR / "numpy_lr_params.npz"

CLASS_NAMES = ["Not satisfied", "Satisfied"]

MODEL_INFO = {
    "MSA baseline - Logistic Regression": {
        "key": "msa_lr",
        "pipeline": "MSA",
        "vectorizer": "msa",
        "test_features": "Xte_m",
        "description": "Logistic Regression trained on MSA baseline preprocessing."
    },
    "MSA baseline - SVM": {
        "key": "msa_svm",
        "pipeline": "MSA",
        "vectorizer": "msa",
        "test_features": "Xte_m",
        "description": "Linear SVM trained on MSA baseline preprocessing."
    },
    "Egyptian - Logistic Regression (sklearn)": {
        "key": "egy_lr",
        "pipeline": "Egyptian",
        "vectorizer": "egy",
        "test_features": "Xte_e",
        "description": "Scikit-learn Logistic Regression with Egyptian preprocessing."
    },
    "Egyptian - SVM": {
        "key": "egy_svm",
        "pipeline": "Egyptian",
        "vectorizer": "egy",
        "test_features": "Xte_e",
        "description": "Linear SVM with Egyptian preprocessing."
    },
    "Egyptian - Voting (Logistic Regression + SVM + NumPy scratch)": {
        "key": "egy_voting",
        "pipeline": "Egyptian",
        "vectorizer": "egy",
        "test_features": "Xte_e",
        "description": "Voting ensemble combining Logistic Regression, SVM and Naive Bayes."
    },
    "Egyptian - Logistic Regression (NumPy scratch)": {
        "key": "numpy_lr",
        "pipeline": "Egyptian",
        "vectorizer": "egy",
        "test_features": "Xte_e",
        "description": "Logistic Regression implemented from scratch using NumPy."
    }
}


@st.cache_resource
def load_artifacts():
    artifacts = joblib.load(ARTIFACTS_PATH)

    params = np.load(NUMPY_PARAMS_PATH)

    numpy_params = {
        "w": params["w"].reshape(-1),
        "b": float(params["b"].reshape(-1)[0])
    }

    return artifacts, numpy_params


artifacts, numpy_params = load_artifacts()

models = artifacts["models"]
vectorizers = artifacts["vectorizers"]
test_data = artifacts["test_data"]
configs = artifacts["preprocessing_config"]


def numpy_predict(X):
    w = numpy_params["w"]
    b = numpy_params["b"]

    scores = np.asarray(X @ w).reshape(-1) + b
    scores = np.clip(scores, -500, 500)

    probabilities = 1 / (1 + np.exp(-scores))

    return (probabilities >= 0.5).astype(int)


def predict_model(model_name, X):
    if model_name == "Egyptian - Logistic Regression (NumPy scratch)":
        return numpy_predict(X)

    model_key = MODEL_INFO[model_name]["key"]

    return models[model_key].predict(X)


def predict_text(model_name, text):
    info = MODEL_INFO[model_name]

    if info["pipeline"] == "MSA":
        cfg = configs["msa_cfg"]
    else:
        cfg = configs["egy_cfg"]

    cleaner = make_cleaner(cfg)

    cleaned_text = cleaner(text)

    vectorizer = vectorizers[info["vectorizer"]]

    features = vectorizer.transform([cleaned_text])

    prediction = predict_model(model_name, features)

    return int(prediction[0]), cleaned_text



def get_test_predictions(model_name):
    info = MODEL_INFO[model_name]

    X_test = test_data[info["test_features"]]

    return predict_model(model_name, X_test)


def evaluate_model(model_name):
    y_true = test_data["y_te"]
    y_pred = get_test_predictions(model_name)

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


@st.cache_data
def get_all_results():
    results = []

    for model_name in MODEL_INFO:
        metrics = evaluate_model(model_name)

        results.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Macro-F1": metrics["macro_f1"],
            "Precision (Not satisfied)": metrics["precision"][0],
            "Recall (Not satisfied)": metrics["recall"][0],
            "F1 (Not satisfied)": metrics["f1"][0],
            "Precision (Satisfied)": metrics["precision"][1],
            "Recall (Satisfied)": metrics["recall"][1],
            "F1 (Satisfied)": metrics["f1"][1]
        })

    return pd.DataFrame(results)




st.set_page_config(
    page_title="Part_C",
    page_icon="",
    layout="wide"
)


st.divider()


#sidebar

st.sidebar.title("Model Configuration")

pipeline = st.sidebar.selectbox(
    "",
    ["Egyptian", "MSA"]
)

available_models = [
    name for name, info in MODEL_INFO.items()
    if info["pipeline"] == pipeline
]

model_name = st.sidebar.selectbox(
    "",
    available_models
)

#st.sidebar.write(MODEL_INFO[model_name]["description"])



prediction_tab, evaluation_tab, comparison_tab = st.tabs([
    "Sentiment Prediction",
    "Model Evaluation",
    "Model Comparison"
])



with prediction_tab:

    st.header("Enter Your Text : ")

    review = st.text_area(
        "",
        height=150
    )

    if st.button("Analyze Review", type="primary"):

        if not review.strip():
            st.warning("Please enter a review first.")

        else:
            try:
                prediction, cleaned_text = predict_text(
                    model_name,
                    review
                )

                st.subheader("Prediction Result")

                if prediction == 1:
                    st.success("Satisfied (Positive Review)")
                else:
                    st.error("Not satisfied (Negative Review)")


            except Exception as e:
                st.error(f"Prediction error: {e}")



#evaluation_tab

with evaluation_tab:

    st.header("Model Evaluation")

    st.write(f"Evaluating: **{model_name}**")

    if st.button("Run Evaluation", type="primary"):

        try:
            with st.spinner("Evaluating model on the test set..."):

                metrics = evaluate_model(model_name)

            col1, col2 = st.columns(2)

            col1.metric(
                "Accuracy",
                f"{metrics['accuracy'] * 100:.2f}%"
            )

            col2.metric(
                "Macro-F1",
                f"{metrics['macro_f1'] * 100:.2f}%"
            )

            st.subheader("Class Metrics")

            class_metrics = pd.DataFrame(
                {
                    "Class": CLASS_NAMES,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1"],
                    

                }
            )

            st.dataframe(
                class_metrics.style.format({
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-Score": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots(figsize=(6, 5))

            display = ConfusionMatrixDisplay(
                confusion_matrix=metrics["confusion_matrix"],
                display_labels=CLASS_NAMES
            )

            display.plot(
                ax=ax,
                cmap="Blues",
                colorbar=False
            )

            ax.set_title(model_name)

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Evaluation error: {e}")


#comparison_tab

with comparison_tab:

    st.header("Compare All Models")

    if st.button("Compare Models", type="primary"):

        try:
            with st.spinner("Evaluating all models..."):

                results_df = get_all_results()

            results_df = results_df.sort_values(
                by="Macro-F1",
                ascending=False
            ).reset_index(drop=True)

            st.subheader("Performance Summary")

            st.dataframe(
                results_df.style.format({
                    col: "{:.4f}"
                    for col in results_df.columns
                    if col != "Model"
                }),
                use_container_width=True,
                hide_index=True
            )

            best_model = results_df.iloc[0]

            st.success(
                f"Best model by Macro-F1: {best_model['Model']} "
                f"({best_model['Macro-F1']:.4f})"
            )

            st.subheader("Accuracy Comparison")

            chart_data = results_df.set_index("Model")[[
                "Accuracy",
                "Macro-F1"
            ]]

            st.bar_chart(chart_data)

            csv = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Comparison CSV",
                data=csv,
                file_name="model_comparison.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Comparison error: {e}")


