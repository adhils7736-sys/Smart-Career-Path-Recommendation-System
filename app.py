import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Career Path Recommender", layout="centered")

# Load the saved model components
@st.cache_resource
def load_model():
    try:
        model = joblib.load("career_model.joblib")
        scaler = joblib.load("scaler.joblib")
        le = joblib.load("label_encoder.joblib")
        return model, scaler, le
    except FileNotFoundError:
        st.error("Model files not found. Please ensure .joblib files are in the directory.")
        return None, None, None

model, scaler, le = load_model()

st.title("🎓 Student Career Path Recommendation")
st.write("Enter student details below to predict the best career path.")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        part_time_job = st.selectbox("Has Part-time Job?", [False, True])
        extracurricular = st.selectbox("Extracurricular Activities?", [False, True])
        study_hours = st.number_input("Weekly Self Study Hours", min_value=0, max_value=100, value=15)

    with col2:
        math = st.slider("Math Score", 0, 100, 80)
        history = st.slider("History Score", 0, 100, 80)
        physics = st.slider("Physics Score", 0, 100, 80)
        chemistry = st.slider("Chemistry Score", 0, 100, 80)
        biology = st.slider("Biology Score", 0, 100, 80)
        english = st.slider("English Score", 0, 100, 80)
        geography = st.slider("Geography Score", 0, 100, 80)

    submit = st.form_submit_button("Predict Career Path")

if submit and model:
    # Prepare input data [cite: 19, 20]
    # In the notebook, object columns were label encoded.
    # Usually, for Streamlit, we manually map these if there's only one LabelEncoder object.
    # For this specific dataset: gender (male=1, female=0), bools (True=1, False=0)
    
    input_data = pd.DataFrame({
        'gender': [1 if gender == "male" else 0],
        'part_time_job': [1 if part_time_job else 0],
        'extracurricular_activities': [1 if extracurricular else 0],
        'weekly_self_study_hours': [study_hours],
        'math_score': [math],
        'history_score': [history],
        'physics_score': [physics],
        'chemistry_score': [chemistry],
        'biology_score': [biology],
        'english_score': [english],
        'geography_score': [geography]
    })

    # Scale the features 
    input_scaled = scaler.transform(input_data)

    # Predict [cite: 24]
    prediction = model.predict(input_scaled)
    career_name = le.inverse_transform(prediction)[0]

    st.success(f"### Recommended Career: {career_name}")
    
    # Optional: Show prediction probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)
        st.write("#### Confidence Level:")
        st.progress(float(np.max(probs)))