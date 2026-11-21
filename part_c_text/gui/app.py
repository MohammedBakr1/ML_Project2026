
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from preprocessing import make_cleaner


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

ARTIFACTS_PATH = DATASET_DIR / "artifacts.joblib"
NUMPY_PARAMS_PATH = DATASET_DIR / "numpy_lr_params.npz"
RESULTS_PATH = DATASET_DIR / "egy_test_results_summary.csv"
CONFUSION_MATRIX_PATH = DATASET_DIR / "egy_confusion_matrices.png"

CLASS_NAMES = ["Not satisfied", "Satisfied"]

MODEL_INFO = {
    "MSA baseline - Logistic Regression": {
        "key": "msa_lr",
        "pipeline": "MSA",
        "vectorizer": "msa",
        "result_name": "MSA baseline - LR",
    },
    "MSA baseline - SVM": {
        "key": "msa_svm",
        "pipeline": "MSA",
        "vectorizer": "msa",
        "result_name": "MSA baseline - SVM",
    },
    "Egyptian - Logistic Regression (sklearn)": {
        "key": "egy_lr",
        "pipeline": "Egyptian",
        "vectorizer": "egy",
        "result_name": "Egyptian - LR (sklearn)",
    },
    "Egyptian - SVM": {
        "key": "egy_svm",
        "pipeline": "Egyptian",
        "vectorizer": "egy",
        "result_name": "Egyptian - Linear SVM",
    },
    "Egyptian - Voting (Logistic Regression + SVM + Naive Bayes)": {
        "key": "egy_voting",
        "pipeline": "Egyptian",
        "vectorizer": "egy",
        "result_name": "Egyptian - Voting (LR+SVM+NB)",
    },
    "Egyptian - Logistic Regression (NumPy scratch)": {
        "key": "numpy_lr",
        "pipeline": "Egyptian",
        "vectorizer": "egy",
        "result_name": "Egyptian - LR (NumPy scratch)",
    },
}

CONFUSION_MATRICES = {
    "msa_lr": [
        [1699, 372],
        [272, 3092]
    ],
    "msa_svm": [
        [1701, 370],
        [267, 3097]
    ],
    "egy_lr": [
        [1717, 354],
        [265, 3099]
    ],
    "egy_svm": [
        [1717, 354],
        [263, 3101]
    ],
    "egy_voting": [
        [1722, 349],
        [262, 3102]
    ],
    "numpy_lr": [
        [1707, 364],
        [257, 3107]
    ]
}


@st.cache_resource
def load_artifacts():
    artifacts = joblib.load(ARTIFACTS_PATH)
    params = np.load(NUMPY_PARAMS_PATH)

    numpy_params = {
        "w": params["w"].reshape(-1),
        "b": float(params["b"].reshape(-1)[0]),
    }

    return artifacts, numpy_params


@st.cache_data
def load_saved_results():
    return pd.read_csv(RESULTS_PATH)


artifacts, numpy_params = load_artifacts()

models = artifacts["models"]
vectorizers = artifacts["vectorizers"]
configs = artifacts["preprocessing_config"]


def numpy_predict(X):
    w = numpy_params["w"]
    b = numpy_params["b"]

    scores = np.asarray(X @ w).reshape(-1) + b
    scores = np.clip(scores, -500, 500)

    probabilities = 1 / (1 + np.exp(-scores))

    return (probabilities >= 0.5).astype(int)


def predict_model(model_name, X):
    model_key = MODEL_INFO[model_name]["key"]

    if model_key == "numpy_lr":
        return numpy_predict(X)

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


def get_saved_model_result(model_name, results_df):
    result_name = MODEL_INFO[model_name]["result_name"]

    row = results_df[results_df["Model"] == result_name]

    if row.empty:
        raise ValueError(f"Saved results do not contain: {result_name}")

    return row.iloc[0]


def display_confusion_matrix(model_name):

    model_key = MODEL_INFO[model_name]["key"]
    cm = np.array(CONFUSION_MATRICES[model_key])

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"Confusion Matrix - {model_name}")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=13
            )

    fig.colorbar(image, ax=ax)

    st.pyplot(fig)
    plt.close(fig)


st.set_page_config(
    page_title="Part C",
    page_icon="",
    layout="wide",
)

st.title("Arabic Customer Reviews - Sentiment Analysis")

st.divider()


# Sidebar

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


# Tabs

prediction_tab, evaluation_tab, comparison_tab = st.tabs([
    "Sentiment Prediction",
    "Model Evaluation",
    "Model Comparison"
])


# Sentiment Prediction

with prediction_tab:

    st.header("Enter Your Text")

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


# Model Evaluation

with evaluation_tab:

    st.header("Model Evaluation")

    try:
        results_df = load_saved_results()

        metrics = get_saved_model_result(
            model_name,
            results_df
        )

        st.write(f"Model: **{model_name}**")

        col1, col2 = st.columns(2)

        col1.metric(
            "Accuracy",
            f"{metrics['Accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "Macro-F1",
            f"{metrics['Macro-F1'] * 100:.2f}%"
        )

        st.subheader("Class Metrics")

        class_metrics = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Precision": [
                metrics["P(neg)"],
                metrics["P(pos)"]
            ],
            "Recall": [
                metrics["R(neg)"],
                metrics["R(pos)"]
            ],
            "F1-Score": [
                metrics["F1(neg)"],
                metrics["F1(pos)"]
            ],
        })

        st.dataframe(
            class_metrics.style.format({
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Confusion Matrices")

        display_confusion_matrix(model_name)

    
    except FileNotFoundError as e:
        st.error(f"Required results file was not found: {e.filename}")

    except Exception as e:
        st.error(f"Evaluation display error: {e}")


# Model Comparison

with comparison_tab:

    st.header("Compare All Models")

    try:
        results_df = load_saved_results()

        display_columns = [
            "Model",
            "Accuracy",
            "Macro-F1",
            "P(neg)",
            "R(neg)",
            "F1(neg)",
            "P(pos)",
            "R(pos)",
            "F1(pos)",
        ]

        results_df = results_df[display_columns].sort_values(
            by="Macro-F1",
            ascending=False
        ).reset_index(drop=True)

        st.subheader("Performance Summary")

        st.dataframe(
            results_df.style.format({
                col: "{:.4f}"
                for col in display_columns
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

        st.subheader("Accuracy and Macro-F1 Comparison")

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

        st.subheader("Confusion Matrices")
        
        if CONFUSION_MATRIX_PATH.exists():
        
            st.image(
                str(CONFUSION_MATRIX_PATH),
                use_container_width=True
            )

    except FileNotFoundError as e:
        st.error(f"Required results file was not found: {e.filename}")

    except Exception as e:
        st.error(f"Comparison display error: {e}")
