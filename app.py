import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

#loading the model
model=joblib.load('model_filename.pkl')
st.title("Student performance ")
st.write("Age: The age of the students ranges from 15 to 18 years.")
st.write("Gender: Gender of the students, where 0 represents Male and 1 represents Female.")
st.write("Ethnicity: 0,Caucasian,1:African American,2:Asian,3:Other")
st.write("StudyTimeWeekly: Weekly study time in hours, ranging from 0 to 20.")
st.write("Tutoring: Tutoring status, where 0 indicates No and 1 indicates Yes.")
st.write("ParentalSupport: The level of parental support, coded as follows,0:None,1:Low,2:Moderate,3:High,4:very high")
st.write("ParentalEducation: The education level of the parents, coded as follows:0:None,1:High School,2: Some College,:3:Bachelor's,4:higher")
st.write("Extracurricular: Participation in extracurricular activities, where 0 indicates No and 1 indicates Yes.")
st.write("Sports: Participation in sports, where 0 indicates No and 1 indicates Yes.")
st.write("Music: Participation in music activities, where 0 indicates No and 1 indicates Yes.")
st.write("Volunteering: Participation in volunteering, where 0 indicates No and 1 indicates Yes.")
st.write("Absences: Number of absences during the school year, ranging from 0 to 30.")
st.write("Classification of students' grades,0:'A',1:'B',2:'C',3: 'D', 4: 'F'  ")


age = st.selectbox("Age", [15,16,17,18])
Gender = st.selectbox("	Gender", [0,1])
ethnicity = st.selectbox("Ethnicity", [0,1,2,3])
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



