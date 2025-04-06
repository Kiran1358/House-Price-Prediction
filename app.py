import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import pandas as pd

# Load model components
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.markdown("Enter house details to predict its selling price.")

# Manual inputs
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 500, 10000, step=50)
OverallQual = st.slider("Overall Quality (1-10)", 1, 10)
YearBuilt = st.number_input("Year Built", 1900, 2025)

# Create input vector
input_data = {"GrLivArea": GrLivArea, "OverallQual": OverallQual, "YearBuilt": YearBuilt}
user_vector = np.zeros(len(features))
for i, col in enumerate(features):
    if col in input_data:
        user_vector[i] = input_data[col]

user_vector_scaled = scaler.transform([user_vector])

if st.button("Predict Price"):
    prediction = model.predict(user_vector_scaled)
    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")

# Optional: Visualize
st.markdown("---")
st.markdown("### 📊 Visualizations")
if st.checkbox("Show Price vs Area Chart"):
    data = pd.read_csv("train.csv")
    fig = px.scatter(data, x="GrLivArea", y="SalePrice", color="OverallQual", title="Price vs Area")
    st.plotly_chart(fig)