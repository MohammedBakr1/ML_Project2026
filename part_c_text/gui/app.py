
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt

from preprocessing import clean


st.set_page_config(
    page_title="Egyptian Arabic Sentiment Analysis_Part C",
    page_icon="",
    layout="centered"
)



BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "saved_models"


@st.cache_resource
def load_models():

    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

    models = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.pkl"
        ),
        "Linear SVM": joblib.load(
            MODEL_DIR / "linear_svm.pkl"
        ),
        "Voting Ensemble": joblib.load(
            MODEL_DIR / "voting_ensemble.pkl"
        )  
    }

    return vectorizer, models


@st.cache_resource
def load_test_data():

    data = joblib.load(MODEL_DIR / "test_data.pkl")

    return data["X_test"], data["y_test"]

vectorizer, models = load_models()

X_test, y_test = load_test_data()

st.title("Part C")


st.divider()

st.subheader("Choose Model")

selected_model = st.selectbox(
    "Select a classification model:",
    list(models.keys())
)

review = st.text_area(
    "Enter Your Text: ",
    height=150
)


if st.button("Enter", type="primary"):

    if not review.strip():

        st.warning("Please enter a text")

    else:

        cleaned_review = clean(review)

        if not cleaned_review.strip():
            st.warning("Not avaliable text")

        else:
            features = vectorizer.transform([cleaned_review])

            model = models[selected_model]

            prediction = model.predict(features)[0]

            st.session_state["prediction"] = (
                selected_model,
                "Positive" if prediction == 1 else "Negative"
            )



if "prediction" in st.session_state:
    model_name, result = st.session_state["prediction"]

    st.subheader("Prediction Result")
    st.write(f"Model: {model_name}")

    if result == "Positive":
        st.success("Positive Review")
    else:
        st.error("Negative Review")


st.divider()


st.header("Project Information")

col1, col2, col3 = st.columns(3)

with col1:
    evaluation_btn = st.button("Model Evaluation", use_container_width=True)


with col2:
    comparison_btn = st.button("Model Comparison", use_container_width=True)


with col3:
    confusion_btn = st.button("Confusion Matrices", use_container_width=True)



if evaluation_btn:
    st.session_state["section"] = "evaluation"

if comparison_btn:
    st.session_state["section"] = "comparison"

if confusion_btn:
    st.session_state["section"] = "confusion"

st.divider()

section = st.session_state.get("section", None)



if section == "evaluation":

    st.subheader("Model Evaluation")

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": __import__("sklearn.metrics", fromlist=["precision_score"]).precision_score(y_test, y_pred, average="macro"),
            "Recall": __import__("sklearn.metrics", fromlist=["recall_score"]).recall_score(y_test, y_pred, average="macro"),
            "Macro F1": f1_score(y_test, y_pred, average="macro"),
            "Weighted F1": f1_score(y_test, y_pred, average="weighted")
        })

    evaluation_data = pd.DataFrame(results)

    st.dataframe(
        evaluation_data,
        use_container_width=True,
        hide_index=True
    )



elif section == "comparison":

    st.subheader("Model Comparison")

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy (%)": accuracy_score(y_test, y_pred) * 100,
            "Macro F1": f1_score(y_test, y_pred, average="macro"),
            "Weighted F1": f1_score(y_test, y_pred, average="weighted")
        })

    comparison_data = pd.DataFrame(results)

    st.bar_chart(
        comparison_data.set_index("Model")[["Accuracy (%)"]]
    )

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )


elif section == "confusion":

    st.subheader("Confusion Matrices")

    for name, model in models.items():

        y_pred = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        st.markdown(f"### {name}")

        fig, ax = plt.subplots(figsize=(6, 4))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Negative", "Positive"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )

        st.pyplot(fig)
        plt.close(fig)


