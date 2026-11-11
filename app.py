import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("iris_model.pkl")

model = load_model()

st.title("🌸 Iris Flower Classification")
st.write(
    "Enter the four flower measurements below. "
    "The trained Random Forest model will predict the Iris species."
)

st.subheader("Flower measurements (cm)")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal length", min_value=0.0, max_value=10.0,
        value=5.1, step=0.1
    )
    petal_length = st.number_input(
        "Petal length", min_value=0.0, max_value=10.0,
        value=1.4, step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal width", min_value=0.0, max_value=10.0,
        value=3.5, step=0.1
    )
    petal_width = st.number_input(
        "Petal width", min_value=0.0, max_value=10.0,
        value=0.2, step=0.1
    )

if st.button("Predict flower species", type="primary", use_container_width=True):
    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = int(model.predict(features)[0])
    species = ["setosa", "versicolor", "virginica"][prediction]

    st.success(f"Predicted species: **{species.capitalize()}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        st.subheader("Model probability scores")
        for name, probability in zip(
            ["Setosa", "Versicolor", "Virginica"], probabilities
        ):
            st.write(f"{name}: {probability:.1%}")
            st.progress(float(probability))

st.caption(
    "Educational demonstration. Predictions depend on the model's training data "
    "and should not be treated as botanical certification."
)
