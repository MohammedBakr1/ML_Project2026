import streamlit as st

st.set_page_config(
    page_title="Parrt_c_test",
    layout="wide",
    initial_sidebar_state="collapsed"

)

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #F5F7FB;
}

/* Remove default top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Header */
.main-header {
    text-align: center;
    padding: 20px 0 15px 0;
}

.main-header h1 {
    font-size: 38px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 8px;
}

.main-header p {
    color: #64748B;
    font-size: 16px;
}

/* Section titles */
.section-title {
    color: #1E293B;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 12px;
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
}

/* Buttons */
div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    background-color: #4F46E5;
    color: white;
    font-weight: 700;
    border: none;
    transition: 0.2s;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #4338CA;
    color: white;
}

/* Text area */
textarea {
    border-radius: 10px !important;
}

/* Model description */
.model-info {
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    padding: 14px;
    border-radius: 10px;
    color: #3730A3;
    font-size: 14px;
}

/* Result placeholder */
.result-box {
    background: #F8FAFC;
    border: 1px dashed #CBD5E1;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    color: #64748B;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 13px;
    padding-top: 25px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1>Part C</h1>
    <p>Just Test</p>
</div>
""", unsafe_allow_html=True)

st.divider()



st.markdown('<p class="section-title">Model Configuration</p>',
            unsafe_allow_html=True)


with st.container(border=True):

    model_choice = st.radio(
        "Select your prediction model",
        [
            "Logistic Regression",
            "Linear SVM",
            "Ensemble"
        ],
        horizontal=True
    )



st.write("")


st.markdown('<p class="section-title">Customer Review</p>',
            unsafe_allow_html=True)

with st.container(border=True):

    review = st.text_area(
        "Enter your customer review below:",
        placeholder="Example: The product quality is amazing. I really enjoyed using it!",
        height=180,
        max_chars=5000,
        key="review_input"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.caption(f"Characters: {len(review)} / 5000")

    with col2:
        st.caption(f"Words: {len(review.split())}")

    st.progress(min(len(review) / 5000, 1.0))


st.write("")


predict_clicked = st.button(
    "Predict Sentiment",
    use_container_width=True
)

if predict_clicked:

    if not review.strip():

        st.warning("Please enter a customer review first.")

    else:

        st.info(
            f"""
            **برضو ده تجربة انت بجد مستني نتيجة ؟؟؟**

            Selected Model: `{model_choice}`

            """
        )