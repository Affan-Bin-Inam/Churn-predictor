import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

API_URL = "http://127.0.0.1:8000/predict"

st.title("📊 Customer Churn Predictor")
st.markdown("Enter a customer's details to estimate their likelihood of churning.")

# input fields
# gender = st.selectbox("Gender", options=['Male', 'Female'])
# SeniorCitizen = st.radio('Senior Citizen', options=['Yes', 'No'])
# Partner = st.radio('Parter', options=['Yes', 'No'])
# Dependents = st.radio('Dependents', options=['Yes', 'No'])
# PhoneService = st.radio('Phone Service', options=['Yes', 'No'])
# MultipleLines = st.selectbox('Multiple Lines', options=['Yes', 'No', 'No Phone Service'])
# InternetService = st.selectbox('Internet Service', options=['DSL', 'Fiber optic', 'No'])
# OnlineSecurity  = st.radio('Online Security', options=['Yes', 'No'])
# OnlineBackup = st.radio('Online Backup', options=['Yes', 'No'])
# DeviceProtection = st.radio('Device Protection', options=['Yes', 'No'])
# TechSupport = st.radio('Tech Support', options=['Yes', 'No'])
# StreamingTV = st.radio('Streaming TV', options=['Yes', 'No'])
# StreamingMovies = st.radio('Streaming Movies', options=['Yes', 'No'])
# Contract = st.selectbox('Contract', options=['Month-to-month', 'One year', 'Two year'])
# PaperlessBilling = st.radio('Paperless Billing', options=['Yes', 'No'])
# PaymentMethod = st.selectbox('Payment Method', options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
#        'Credit card (automatic)'])
# Tenure = st.number_input('Tenure', min_value= 0, value= 0)
# MonthlyCharges = st.number_input('Monthly Charges', min_value= 0, value= 0)



# if st.button('Pridict!!'):
#     input_data = {
#         'gender': gender,
#         'SeniorCitizen': SeniorCitizen,
#         'Partner': Partner,
#         'Dependents': Dependents,
#         'PhoneService': PhoneService,
#         'MultipleLines': MultipleLines,
#         'InternetService': InternetService,
#         'OnlineSecurity': OnlineSecurity,
#         'OnlineBackup': OnlineBackup,
#         'DeviceProtection': DeviceProtection,
#         'TechSupport': TechSupport,
#         'StreamingTV': StreamingTV,
#         'StreamingMovies': StreamingMovies,
#         'Contract': Contract,
#         'PaperlessBilling': PaperlessBilling,
#         'PaymentMethod': PaymentMethod,
#         'Tenure': Tenure,
#         'MonthlyCharges': MonthlyCharges  
#         }

    # try:
    #     response = requests.post(API_URL, json=input_data)
    #     if response.status_code == 200:
    #         result = response.json()
    #         st.success(f"The verdict on, if the customer is likel to churn is: **{result['churn_prediction']}** \nWith Probability: **{result['churn_probablility']}**")
    #     else:
    #         st.error(f"API error: {response.status_code} - {response.text}")
    # except requests.exceptions.ConnectionError:
    #     st.error("Couldn't Connect to FastAPI Server, Make sure it's running on port 8000")


with st.form("churn_form"):
    st.subheader("Customer Profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", options=['Male', 'Female'])
        senior_input = st.radio('Senior Citizen', options=['Yes', 'No'])
        Partner = st.radio('Partner', options=['Yes', 'No'])
        Dependents = st.radio('Dependents', options=['Yes', 'No'])
        Tenure = st.number_input('Tenure (months)', min_value=0, value=0)
    with col2:
        Contract = st.selectbox('Contract', options=['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.radio('Paperless Billing', options=['Yes', 'No'])
        PaymentMethod = st.selectbox('Payment Method', options=[
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        MonthlyCharges = st.number_input('Monthly Charges ($)', min_value=0.0, value=0.0, step=0.01)

    st.subheader("Services")
    col3, col4 = st.columns(2)
    with col3:
        PhoneService = st.radio('Phone Service', options=['Yes', 'No'])
        MultipleLines = st.selectbox('Multiple Lines', options=['Yes', 'No', 'No phone service'])
        InternetService = st.selectbox('Internet Service', options=['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.radio('Online Security', options=['Yes', 'No'])
    with col4:
        OnlineBackup = st.radio('Online Backup', options=['Yes', 'No'])
        DeviceProtection = st.radio('Device Protection', options=['Yes', 'No'])
        TechSupport = st.radio('Tech Support', options=['Yes', 'No'])
        StreamingTV = st.radio('Streaming TV', options=['Yes', 'No'])
        StreamingMovies = st.radio('Streaming Movies', options=['Yes', 'No'])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    SeniorCitizen = 1 if senior_input == 'Yes' else 0
    input_data = {
        'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner,
        'Dependents': Dependents, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
        'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup, 'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport, 'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
        'Contract': Contract, 'PaperlessBilling': PaperlessBilling, 'PaymentMethod': PaymentMethod,
        'Tenure': Tenure, 'MonthlyCharges': MonthlyCharges
    }

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            prob = result['churn_probablility']
            if result['churn_prediction']:
                st.error(f"⚠️ Likely to churn — probability: **{prob:.1%}**")
            else:
                st.success(f"✅ Likely to stay — churn probability: **{prob:.1%}**")
            st.progress(prob)
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Couldn't connect to the FastAPI server. Make sure it's running on port 8000.")