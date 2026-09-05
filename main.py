from fastapi import FastAPI, Path, HTTPException, Query
from fastapi.responses import JSONResponse
import json, joblib
# from pydantic import BaseModel, Field, computed_field, field_validator
# from typing import Annotated, Literal
import pandas as pd
from customer_input import CustomerInput


app = FastAPI()

# loading data
model_columns = joblib.load('model_columns.pkl')
model = joblib.load('churn_model.pkl')

# class CustomerInput(BaseModel):
#     gender: Annotated[str, Field(..., description='Select male, female')] 
#     SeniorCitizen: Annotated[Literal[1,0], Field(..., description='1 for yes, 0 for no')]
#     Partner: Annotated[str, Field(..., description='yes or no')]
#     Dependents: Annotated[str, Field(..., description='yes or no')]
#     Tenure: Annotated[int, Field(..., ge=0, description='number of months you have been a member')]
#     PhoneService: Annotated[str, Field(..., description='yes or no')] 
#     MultipleLines: Annotated[str, Field(..., description='no phone service, yes or no')]
#     InternetService: Annotated[Literal['DSL', 'Fiber optic', 'No'], Field(..., description='name the internet service or select no')]
#     OnlineSecurity: Annotated[str, Field(..., description='yes or no')]
#     OnlineBackup: Annotated[str, Field(..., description='yes or no')]
#     DeviceProtection: Annotated[str, Field(..., description='yes or no')]
#     TechSupport: Annotated[str, Field(..., description='yes or no')]
#     StreamingTV: Annotated[str, Field(..., description='yes or no')]
#     StreamingMovies: Annotated[str, Field(..., description='yes or no')]
#     Contract: Annotated[Literal['Month-to-month', 'One year', 'Two year'] , Field(...,description='type of contract')]
#     PaperlessBilling: Annotated[str, Field(..., description='yes or no')]
#     PaymentMethod: Annotated[Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
#        'Credit card (automatic)'], Field(..., description='type of payment method')]
#     MonthlyCharges: Annotated[float, Field(...,gt=0 ,description='enter your monthly amount')]
#     # TotalCharges: Annotated[float, Field(..., description='enter your monthly amount')]


#     @field_validator('OnlineBackup', 'OnlineSecurity', 'PhoneService', 'StreamingMovies', 'StreamingTV', 'TechSupport', 'DeviceProtection', 'Dependents', 'PaperlessBilling')
#     @classmethod
#     def lower_case(cls, value: str) -> str:
#         value = value.lower()

#         if value == 'yes':
#             return 'Yes'
#         if value == 'no':
#             return 'No'
#         else:
#             raise ValueError('Enter only Yes and No')

#     @field_validator('gender')
#     @classmethod
#     def normalize_gender(cls, value: str) -> str:
#         value = value.lower()

#         if value == 'male':
#             return 'Male'
#         elif value == 'female':
#             return 'Female'
#         else:
#             raise ValueError("Gender must be Male or Female")

#     @field_validator('SeniorCitizen', mode='before')
#     @classmethod
#     def normalize_citizen(cls, value) -> int:
#         if isinstance(value, str):
#             if value.lower() == 'yes':
#                 return 1
#             elif value.lower() == 'no':
#                 return 0
#             raise ValueError("Only select Yes or No")
#         return value      
        
#     @computed_field
#     @property
#     def TenureCategory(self) -> str:
#         if self.Tenure <=5:
#             return "New"
#         elif self.Tenure <12:
#             return 'Developing'
#         elif self.Tenure <24:
#             return 'Established'
#         elif self.Tenure <48:
#             return 'Loyal'
#         else:
#             return 'Vetran'
        
                

#     @computed_field
#     @property
#     def TotalServices(self) -> int:
#         total_count = 0
#         if self.OnlineSecurity == 'Yes':
#             total_count +=1
#         if self.OnlineBackup == 'Yes':
#             total_count +=1
#         if self.PhoneService == 'Yes':
#             total_count +=1
#         if self.StreamingMovies == 'Yes':
#             total_count +=1
#         if self.StreamingTV == 'Yes':
#             total_count +=1
#         if self.TechSupport == 'Yes':
#             total_count +=1
#         if self.DeviceProtection == 'Yes':
#             total_count +=1
#         return total_count

#     @computed_field
#     @property
#     def TotalCharges(self) -> float:
#         return self.MonthlyCharges * self.Tenure
        

        
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