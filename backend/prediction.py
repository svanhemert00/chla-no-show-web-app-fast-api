# This program is used at the back-end FastAPI to receive the user inputs from
# Streamlit-based front-end user_inputs.py program via JSON data structure
# It then loads up the random_forest_model.pkl (Prediction ML Model) and label_encoder.pkl
# Invokes the random_forest_model with input data structure to and generates the prediction
# It returns the Prediction to the front-end via FastAPI's @app.post process

### import libraries
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date, datetime
import json
from typing import List

### Instantiate FastAPI
app = FastAPI()

class UserInput(BaseModel):
    start_datetime: str
    end_datetime: str
    clinic_selector: List[str]

file_path = 'CHLA_clean_data_2024_Appointments.csv'
df = pd.read_csv(file_path)
df['APPT_DATE'] = pd.to_datetime(df['APPT_DATE'])

@app.post("/process/")
async def process_item(user_input: UserInput):
    
    start_datetime_str = user_input.start_datetime
    end_datetime_str = user_input.end_datetime
    clinic_selector = user_input.clinic_selector
    
    # Parse strings to datetime objects
    start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%d %H:%M:%S')
    
    ### filter df by date inputs and return caption
    if start_datetime and end_datetime:
        mask = (df['APPT_DATE'] >= start_datetime) & (df['APPT_DATE'] <= end_datetime)
        filtered_df = df[mask]
    
    ### select and filter filtered_df by clinic
    if len(clinic_selector) == 0:  # Check if no clinic is selected
        filtered_df = filtered_df.copy()  # Retain the original DataFrame
    else:
        filtered_df = filtered_df[filtered_df['CLINIC'].isin(clinic_selector)]
        
    ### slice MRN
    fdf = filtered_df[[
        'MRN',
        'APPT_DATE',
        'AGE',
        'CLINIC',
        'TOTAL_NUMBER_OF_CANCELLATIONS',
        'LEAD_TIME',
        'TOTAL_NUMBER_OF_RESCHEDULED',
        'TOTAL_NUMBER_OF_NOSHOW',
        'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT',
        'HOUR_OF_DAY',
        'NUM_OF_MONTH'
    ]]
    
    ### slice predictive df
    pdf = fdf.drop(['MRN', 'APPT_DATE'], axis=1) #! THIS WILL GO IN PREDICTION.PY
    
    ### load and run the predictor model
    model = pickle.load(open('random_forest_model.pkl', 'rb'))

    ### label encoding
    le = LabelEncoder()
    object_cols = ['CLINIC']
    for col in object_cols:
        pdf[col] = le.fit_transform(pdf[col])
        
    ### run model and output predictions   
    predictions = model.predict(pdf)
    predictions_series = pd.Series(predictions)
    fdf = fdf.reset_index(drop=True)
    final_df = pd.concat([fdf, predictions_series], axis=1)
    final_df.columns = [*final_df.columns[:-1], 'NO SHOW (Y/N)']
    final_df = final_df[['MRN', 'APPT_DATE', 'CLINIC', 'NO SHOW (Y/N)']]
    no_show_mapping = {0: 'NO', 1: 'YES'}
    final_df['NO SHOW (Y/N)'] = final_df['NO SHOW (Y/N)'].replace(no_show_mapping)
    final_df['MRN'] = final_df['MRN'].astype(str)
    final_df = final_df.sort_values(by='CLINIC')
    final_df = final_df.sort_values(by='APPT_DATE')
    final_df.rename(columns={'APPT_DATE': 'APPOINTMENT DATE'}, inplace=True)
    for index, row in final_df.iterrows():
        if row['NO SHOW (Y/N)'] == 'Yes':
            final_df.at[index, 'RECOMMENDATION'] = "DOUBLE-BOOK"
        else:
            final_df.at[index, 'RECOMMENDATION'] = "DON'T DOUBLE-BOOK"
    final_df['APPOINTMENT DATE'] = final_df['APPOINTMENT DATE'].astype(str)
    
    # Return the final_df as JSON
    return {'final_df': final_df.to_json()}