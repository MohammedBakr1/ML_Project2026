import streamlit as st

st.set_page_config(
    page_title="Parrt_c_test",
    layout="centered"
)

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 16px;
    margin-bottom: 25px;
}

.arabic-text {
    direction: rtl;
    text-align: right;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">PartC</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


st.subheader("Model Configuration")

model_choice = st.selectbox(
    "Choose a Model",
    [
        "Logistic Regression",
        "Linear SVM",
        "Ensemble"
    ]
)


st.subheader("Customer Review")

review = st.text_area(
    "Enter your text:",
    height=150,
    max_chars=5000
)


if st.button("Predict Sentiment"):

    if not review.strip():

        st.warning("Please enter a review first.")

    else:

        st.info(
            "ده تجربة يا برو انت مستني اجابة ولا ايه ؟؟؟ "
        )



