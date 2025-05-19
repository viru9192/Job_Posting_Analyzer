# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from scipy.sparse import hstack
import joblib

# Suspicious Keyword Detection Function
def count_suspicious_keywords(text):
    keywords = [
        'money', 'quick', 'no experience', 'easy', 'limited time', 'guarantee', 'click here',
        'winner', 'urgent', 'apply now', 'limited spots', 'make money', 'investment opportunity',
        'immediate start', 'work from home', 'crypto', 'bitcoin', 'wire transfer', 'cash reward',
        'get paid', 'no interview', 'sign up bonus', 'unlimited earning', '100% free', 'fast cash'
    ]
    return sum([len(re.findall(rf'\b{re.escape(kw)}\b', text.lower())) for kw in keywords])

# Experience Extractor 
def extract_years_experience(text):
    matches = re.findall(r'(\d+)\s*\+?\s*(?:years?|yrs?)', text.lower())
    return int(matches[0]) if matches else 0

# Load Pretrained TF-IDF and Model 
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("best_model.pkl")

st.title("🔍 Fake Job Posting Detector")

st.markdown("Enter job details below to detect if it's **real or fake** using ML")

with st.form("job_form"):
    title = st.text_input("Job Title")
    description = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")
    profile = st.text_area("Company Profile")
    benefits = st.text_area("Benefits")
    avg_salary = st.number_input("Average Salary", min_value=0)
    is_remote = st.selectbox("Is this a remote job?", ["No", "Yes"])
    years_exp = st.slider("Years of Experience Required", 0, 20, 1)
    submitted = st.form_submit_button("Predict")

if submitted:
    combined_text = f"{title} {description} {requirements} {profile} {benefits}"
    suspicious_count = count_suspicious_keywords(combined_text)
    is_remote_flag = 1 if is_remote == "Yes" else 0
    years_experience = extract_years_experience(requirements + ' ' + description)

    tfidf_vec = tfidf.transform([combined_text])
    numeric_features = pd.DataFrame([[avg_salary, 0, is_remote_flag, suspicious_count, years_exp]])
    final_input = hstack([tfidf_vec, numeric_features])

    pred = model.predict(final_input)[0]
    proba = model.predict_proba(final_input)[0,1]

    if pred == 1:
        st.error(f"🚨 This job posting is predicted to be **FAKE** (Confidence: {proba:.2%})")
    else:
        st.success(f"✅ This job posting is predicted to be **REAL** (Confidence: {1 - proba:.2%})")
