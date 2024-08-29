import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

#loading the model
model=joblib.load('model_filename.pkl')
st.title("Student performance ")

age = st.number_input("Age", min_value=16, max_value=18, value=17, step=1)
Gender = st.number_input("Gender", min_value=0, max_value=1, value=0, step=1)