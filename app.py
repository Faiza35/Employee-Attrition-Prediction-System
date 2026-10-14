"""Streamlit web interface (bonus).  Run: streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import load_model, predict_attrition  # noqa: E402
from utils import DEPARTMENTS, InputValidationError  # noqa: E402

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👔")
st.title("👔 Employee Attrition Prediction")
st.caption("Enter employee details to estimate the probability that they will leave.")


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except (FileNotFoundError, RuntimeError) as err:
    st.error(str(err))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    department = st.selectbox("Department", DEPARTMENTS)
    income = st.number_input("Monthly Income (₹)", min_value=1, value=50000, step=1000)
    years = st.number_input("Years at Company", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    distance = st.number_input("Distance from Home (km)", min_value=0, max_value=500, value=10)
with col2:
    job_sat = st.slider("Job Satisfaction", 1, 5, 3)
    work_life = st.slider("Work-Life Balance", 1, 5, 3)
    performance = st.slider("Performance Rating", 1, 5, 3)
    overtime = st.radio("OverTime", ["No", "Yes"], horizontal=True)

if st.button("Predict", type="primary"):
    employee = {
        "Age": age, "Department": department, "MonthlyIncome": income,
        "YearsAtCompany": years, "JobSatisfaction": job_sat, "WorkLifeBalance": work_life,
        "PerformanceRating": performance, "OverTime": overtime, "DistanceFromHome": distance,
    }
    try:
        result = predict_attrition(model, employee)
    except InputValidationError as err:
        st.error(f"Input error: {err}")
    else:
        prob = result["probability"]
        if result["prediction"] == "Yes":
            st.error(f"⚠️ Likely to LEAVE  —  probability {prob * 100:.1f}%")
        else:
            st.success(f"✅ Likely to STAY  —  probability of leaving {prob * 100:.1f}%")
        st.progress(min(max(prob, 0.0), 1.0))
