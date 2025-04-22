import streamlit as st
import seaborn as sns 
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


@st.cache_resource
def load_model():
    return joblib.load("model/model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("data/startup_data.csv")

model = load_model()
data = load_data()

# st.set_page_config(page_title='Startup Success Indicator',layout='wide')
st.title("🚀 Startup Success Predictor")

st.sidebar.header("📊 Input Startup Details")
company_name = st.sidebar.text_input("Company Name")
industry_options = sorted(data['industry'].dropna().astype(str).unique())
industry = st.sidebar.selectbox('Industry', industry_options)
filtered_data = data[data['industry'] == industry]

st.subheader(f"💰 Funding in {industry}")
st.metric("Median Funding", f"${filtered_data['total_funding_usd'].median():,.0f}")
st.line_chart(filtered_data["total_funding_usd"])


# industry_options = sorted(data['industry'].dropna().astype(str).unique())
# industry = st.sidebar.selectbox('Industry', industry_options)
headquarters_options = sorted(data['headquarters'].astype(str).unique())
headquarters = st.sidebar.selectbox('Headquarters', headquarters_options)
founded_year = st.sidebar.number_input('Founded Year', min_value=2000, max_value=2025, step=1)
funding_amount = st.sidebar.number_input('Total Funding (USD)', min_value=0.0, step=100000.0)
num_rounds = st.sidebar.slider('Number of Funding Rounds', min_value=0, max_value=15, step=1)
investors_count = st.sidebar.slider('Number of Investors', min_value=0, max_value=50, step=1)
founder_experience = st.sidebar.slider('Founder Experience (Years)', min_value=0, max_value=50, step=1)
last_funding_year = st.sidebar.number_input('Last Funding Year', min_value=2000, max_value=2025, step=1)
last_funding_month = st.sidebar.selectbox('Last Funding Month', list(range(1, 13)))

if st.sidebar.button("Predict Success"):
    # Construct input DataFrame
    input_data = pd.DataFrame([{
        'company_name': company_name,
        'industry': industry,
        'headquarters': headquarters,
        'founded_year': founded_year,
        'total_funding_usd': funding_amount,
        'num_funding_rounds': num_rounds,
        'investors_count': investors_count,
        'founder_experience': founder_experience,
        'last_funding_year': last_funding_year,
        'last_funding_month': last_funding_month
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    # prob = model.predict_proba(input_data)
    # class_labels = model.classes_
    # prediction_index = list(class_labels).index(prediction)
    if prediction == 'success':
        st.success(f"🎯 Prediction: Success")
    else:
        st.warning(f"🎯 Prediction: High Risk of Failure")
    # st.write("Prediction output:", model.predict(input_data))
    # st.write(f"Confidence: {prob[prediction_index]*100:.2f}%")

    # Insights section
    st.subheader("💡 Industry Insights")
    industry_data = data[data['industry'] == industry]
    median_funding = industry_data['total_funding_usd'].median()
    median_rounds = industry_data['num_funding_rounds'].median()
    st.write(f"Median Funding in {industry}: ${median_funding:,.0f}")
    st.write(f"Median Funding Rounds: {median_rounds:.2f}")


