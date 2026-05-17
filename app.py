import streamlit as st
import numpy as np
import pickle

# Load Model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit Page Config
st.set_page_config(
    page_title="Random Forest Employee Attrition Prediction",
    layout="centered"
)

# Title
st.title("👨‍💼 Employee Attrition Prediction using Random Forest")

st.write("Enter employee details to predict attrition status.")

# User Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)

salary = st.number_input(
    "Salary",
    min_value=10000,
    max_value=500000,
    value=50000
)

years_company = st.number_input(
    "Years At Company",
    min_value=0,
    max_value=40,
    value=5
)

job_satisfaction = st.slider(
    "Job Satisfaction (1-10)",
    min_value=1,
    max_value=10,
    value=5
)

work_hours = st.number_input(
    "Weekly Work Hours",
    min_value=20,
    max_value=100,
    value=40
)

promotions = st.number_input(
    "Promotions",
    min_value=0,
    max_value=10,
    value=1
)

distance_home = st.number_input(
    "Distance From Home (km)",
    min_value=1,
    max_value=100,
    value=10
)

# Prediction Button
if st.button("Predict Attrition"):

    input_data = np.array([[
        age,
        salary,
        years_company,
        job_satisfaction,
        work_hours,
        promotions,
        distance_home
    ]])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prediction Probability
    probability = model.predict_proba(input_data)[0]

    # Display Result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Employee is Likely to Leave the Company")
    else:
        st.success("Employee is Likely to Stay in the Company")

    st.write(f"Attrition Probability: {probability[1]:.2f}")
    st.write(f"Retention Probability: {probability[0]:.2f}")

st.markdown("---")
