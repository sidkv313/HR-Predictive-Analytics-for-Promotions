# ================ app-2.py ================
import streamlit as st
import joblib
import pandas as pd

# Load artifacts
@st.cache_resource
def load_artifacts():
    return {
        'model': joblib.load('/media/gdeb/Local_Disk/python_rut/hr_model_output/department_encoder.pkl'),
        'encoders': {
            'department': joblib.load('/media/gdeb/Local_Disk/python_rut/hr_model_output/department_encoder.pkl'),
            'education': joblib.load('/media/gdeb/Local_Disk/python_rut/hr_model_output/education_encoder.pkl'),
            'gender': joblib.load('/media/gdeb/Local_Disk/python_rut/hr_model_output/gender_encoder.pkl'),
            'recruitment_channel': joblib.load('/media/gdeb/Local_Disk/python_rut/hr_model_output/recruitment_channel_encoder.pkl')
        }
    }

artifacts = load_artifacts()

# Human-readable options (match your dataset)
DEPARTMENT_OPTIONS = [
    'Sales & Marketing', 'Operations', 'Technology',
    'Analytics', 'R&D', 'Procurement', 
    'Finance', 'HR', 'Legal'
]

EDUCATION_OPTIONS = [
    'Below Secondary', 'High School',
    "Bachelor's", "Master's & above"
]

GENDER_OPTIONS = ['m', 'f']
RECRUITMENT_OPTIONS = ['sourcing', 'other', 'referred']

st.title('🏢 HR Promotion Predictor')
st.markdown("Predict employee promotion likelihood using performance metrics")

# Input widgets
col1, col2 = st.columns(2)

with col1:
    department = st.selectbox('Department', DEPARTMENT_OPTIONS)
    education = st.selectbox('Education Level', EDUCATION_OPTIONS)
    
with col2:
    gender = st.selectbox('Gender', GENDER_OPTIONS)
    recruitment_channel = st.selectbox('Recruitment Channel', RECRUITMENT_OPTIONS)
    age = st.slider('Age', 20, 60, 30)
    avg_training_score = st.slider('Average Training Score', 40, 100, 75)

# Additional inputs
with st.expander("Performance Metrics"):
    no_of_trainings = st.slider('Number of Trainings', 1, 10, 1)
    previous_year_rating = st.selectbox('Previous Year Rating', [1.0, 2.0, 3.0, 4.0, 5.0])
    length_of_service = st.slider('Length of Service (years)', 1, 40, 5)
    kpis_met = st.selectbox('KPIs Met >80%', ['No', 'Yes'])
    awards_won = st.selectbox('Awards Won?', ['No', 'Yes'])

# Prediction logic
if st.button('Predict Promotion'):
    try:
        # Encode categorical features
        department_encoded = artifacts['encoders']['department'].transform([department])[0]
        education_encoded = artifacts['encoders']['education'].transform([education])[0]
        gender_encoded = artifacts['encoders']['gender'].transform([gender])[0]
        recruitment_encoded = artifacts['encoders']['recruitment_channel'].transform([recruitment_channel])[0]
        
        # Convert binary inputs
        kpis_binary = 1 if kpis_met == 'Yes' else 0
        awards_binary = 1 if awards_won == 'Yes' else 0

        # Create input DataFrame
        input_data = pd.DataFrame([{
            'department_encoded': department_encoded,
            'education_encoded': education_encoded,
            'gender_encoded': gender_encoded,
            'recruitment_channel_encoded': recruitment_encoded,
            'no_of_trainings': no_of_trainings,
            'age': age,
            'previous_year_rating': previous_year_rating,
            'length_of_service': length_of_service,
            'KPIs_met >80%': kpis_binary,
            'awards_won?': awards_binary,
            'avg_training_score': avg_training_score
        }])

        # Make prediction
        prediction = artifacts['model'].predict(input_data)[0]
        proba = artifacts['model'].predict_proba(input_data)[0][1]
        
        # Display results
        if prediction == 1:
            st.success(f"🎉 **Promotion Recommended!** (Confidence: {proba*100:.1f}%)")
        else:
            st.warning(f"⏳ **Promotion Not Recommended** (Confidence: {(1-proba)*100:.1f}%)")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Display encoder mappings
with st.expander("🔍 Category Encodings"):
    st.write("Department Encodings:", dict(zip(DEPARTMENT_OPTIONS, artifacts['encoders']['department'].transform(DEPARTMENT_OPTIONS))))
    st.write("Education Encodings:", dict(zip(EDUCATION_OPTIONS, artifacts['encoders']['education'].transform(EDUCATION_OPTIONS))))
