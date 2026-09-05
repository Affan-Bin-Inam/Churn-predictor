from fastapi import FastAPI, Path, HTTPException, Query
from fastapi.responses import JSONResponse
import json, joblib
import pandas as pd
from customer_input import CustomerInput


app = FastAPI()

# loading data
model_columns = joblib.load('model_columns.pkl')
model = joblib.load('churn_model.pkl')

        
# API Code        
@app.post('/predict')
def predict_churn(customer: CustomerInput):
    input_dict = customer.model_dump()
    input_df = pd.DataFrame([input_dict])

    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                         'PhoneService', 'MultipleLines', 'InternetService',
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies',
                         'Contract', 'PaperlessBilling', 'PaymentMethod',
                         'TenureCategory']

    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

    input_final = input_encoded.reindex(columns=model_columns, fill_value=False)
    print(input_final)

    print(input_final.to_dict(orient='records'))
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]

    return JSONResponse(status_code=200, content={'churn_prediction': bool(prediction), 'churn_probablility':float(probability)})