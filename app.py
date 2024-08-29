import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

#loading the model
model=joblib.load('model_filename.pkl')
st.title("Student performance ")

age = st.number_input("Age", min_value=15, max_value=18, value=17, step=1)
Gender = st.number_input("Gender", min_value=0, max_value=1, value=0, step=1)
ethnicity = st.number_input("Ethnicity", min_value=0, max_value=3, value=1, step=1)
study_time_weekly = st.number_input("StudyTimeWeekly", min_value=0, max_value=20, value=10, step=1)
Tutoring= st.selectbox("Tutoring", [0,1, 2])
parental_support = st.selectbox("ParentalSupport", [0,1, 2, 3, 4])
parental_education = st.selectbox("ParentalEducation", [0,1, 2, 3, 4])

Extracurricular = st.selectbox("Extracurricular", [0,1])
Sports = st.selectbox("	Sports", [0,1])
Music = st.selectbox("	Music", [0,1])
Volunteering = st.selectbox("	Volunteering", [0,1])
absences = st.number_input("Absences", min_value=0, max_value=30, value=17, step=1)
Gradeclass = st.selectbox("GradeClass", [0,1,2,3,4])

# Create a button for making predictions
if st.button("Predict"):
    # Process input values into a DataFrame
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [Gender],
            "Ethnicity": [ethnicity],
            "ParentalEducation": [parental_education],
            "StudyTimeWeekly": [study_time_weekly],
            "Absences": [absences],
            "ParentalSupport": [parental_support],
            "Extracurricular": [Extracurricular],
            "Sports": [Sports],
            "Music": [Music],
            "Volunteering": [Volunteering],
            "GradeClass": [Gradeclass],
        }
    )
    
# Preprocess the input data (e.g., encoding categorical variables, scaling, etc.)
            
    scaler=StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.success("A")
    elif prediction[0] == 2:
        st.info("B")
    elif prediction[0] == 3:
        st.warning("C")
    else:
        st.success("D")



