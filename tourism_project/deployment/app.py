"""Streamlit frontend that loads the model from the HF Model Hub."""
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_USERNAME    = "raghavv33"          # <-- change me
MODEL_REPO_ID  = f"{HF_USERNAME}/tourism-model"
MODEL_FILENAME = "best_tourism_model.joblib"

st.set_page_config(page_title="Tourism Package Prediction",
                   page_icon="🌴", layout="centered")
st.title("🌴 Wellness Tourism Package — Purchase Predictor")
st.write("Enter customer details below and the model will predict "
         "whether they are likely to purchase the package.")

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    return joblib.load(path)

model = load_model()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (mins)", 1, 120, 15)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 3)
    num_followups = st.number_input("Number of Followups", 0, 10, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
with col2:
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Number of Trips per Year", 0, 15, 2)
    passport = st.selectbox("Has Passport", [0, 1])
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    own_car = st.selectbox("Owns Car", [0, 1])
    num_children_visiting = st.number_input("Children (<5 yrs) Visiting", 0, 5, 0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 1000.0, 200000.0, 25000.0, step=500.0)

if st.button("Predict"):
    row = pd.DataFrame([{
        "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch, "Occupation": occupation,
        "Gender": gender, "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups, "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status, "NumberOfTrips": num_trips,
        "Passport": passport, "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car, "NumberOfChildrenVisiting": num_children_visiting,
        "Designation": designation, "MonthlyIncome": monthly_income,
    }])
    pred  = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0, 1])
    st.subheader("Prediction")
    if pred == 1:
        st.success(f"✅ Likely to purchase the package (probability = {proba:.2%})")
    else:
        st.warning(f"❌ Unlikely to purchase the package (probability = {proba:.2%})")
    st.caption("Threshold = 0.5 on predicted probability.")
