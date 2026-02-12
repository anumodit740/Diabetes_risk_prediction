import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    return model

model = load_model()

# ----------------------------
# Title
# ----------------------------
st.title("🩺 Diabetes Risk Assessment")
st.markdown("This tool estimates diabetes risk using machine learning.")
st.markdown("⚠️ *Educational purposes only. Not a medical diagnosis.*")

st.divider()

# ----------------------------
# User Input
# ----------------------------
st.subheader("Enter Patient Details")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

st.divider()

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Diabetes Risk"):
    
    try:
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        
        prediction = model.predict(input_data)[0]
        
        # Safe probability handling
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = None
        
        st.subheader("Prediction Result")
        
        if prediction == 1:
            if probability:
                st.error(f"High Risk of Diabetes ({probability*100:.2f}% probability)")
            else:
                st.error("High Risk of Diabetes")
        else:
            if probability:
                st.success(f"Low Risk of Diabetes ({probability*100:.2f}% probability)")
            else:
                st.success("Low Risk of Diabetes")

        if probability:
            st.progress(int(probability * 100))

    except Exception as e:
        st.error(f"Error occurred: {e}")
