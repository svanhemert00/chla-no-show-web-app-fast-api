### import libraries
import numpy as np
import pandas as pd
import streamlit as st
import requests
from datetime import date, datetime
from titlecase import titlecase
import json

def main():
    
    ### Front-end Streamlit segment to prompt and collect user inputs

    ### set page configuration
    st.set_page_config(page_icon="childrens-hospital-la-icon.jpg")


    ### set titles
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.image('childrens-hospital-la-logo.png')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-family: Geneva; color: #005b98;'>Appointment No-Shows Prediction Tool</h3>", unsafe_allow_html=True)
    st.caption("Welcome to the Appointment No-Show Prediction Model Web App for Children's Hospital Los Angeles (CHLA)! We understand that missed appointments can disrupt schedules and delay care for our young patients. That's why we've developed a cutting-edge predictive model to help identify the likelihood of appointment no-shows in advance. Our web app utilizes advanced machine learning algorithms trained on historical data to predict the probability of a patient missing their scheduled appointment.")

    ### ingest data
    @st.cache_resource
    
    def load_data(file_path):
        return pd.read_csv(file_path)

    df = load_data("CHLA_clean_data_2024_Appointments.csv")
    df['APPT_DATE'] = pd.to_datetime(df['APPT_DATE'])
   
    ### date inputs
    col1, col2 = st.columns([1,1])

    with col1:
        start_datetime = st.date_input("Choose Start Date", min_value=df['APPT_DATE'].min(), max_value=df['APPT_DATE'].max())
    with col2:
        end_datetime = st.date_input("Choose End Date", min_value=df['APPT_DATE'].min(), max_value=df['APPT_DATE'].max())

    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)

    if start_datetime > end_datetime:
        st.error("End Date should be after Start Date")
    elif start_datetime < end_datetime:
        start_date = start_datetime.date()
        end_date = end_datetime.date()
        st.caption(f"You have selected appointments between {start_date} and {end_date}")

    ### select and filter filtered_df by clinic
    clinic_selector = st.multiselect("Select a Clinic", df['CLINIC'].unique())
    
    clinic_strings = []
    for i in range(len(clinic_selector)):
        clinic_strings.append(titlecase(str(clinic_selector[i])))
    clinic_string = ", ".join(clinic_strings)
    st.caption(f"You have selected {clinic_string}")
        
    ### Backend API URL (FastAPI)
    # 127.0.0.1: is the loopback IP address, also known as localhost. 
    # It's used to specify that the server is running on the same machine as the client
    # 8000:is the port number where the FastAPI application is running
    # /process/:is the path that specifies a specific endpoint or route in the FastAPI application
    url = "http://127.0.0.1:8000/process/"
    
    start_date_str = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
    end_date_str = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    ### run model button
    run_button = st.button('Run')
    if run_button:

        st.info("Your model is running...") 

        data = {
            'start_datetime': start_date_str,
            'end_datetime': end_date_str,
            'clinic_selector': clinic_selector
        }   
    
        # Send a request to the FastAPI backend and receive Response
        response = requests.post(url, json=data)
            
        # After receiving the response from back-end API
        if response.status_code == 200:
            
            st.success('Run Complete')
            final_df_json = response.json()['final_df']
            final_df = pd.read_json(final_df_json)
            st.write(final_df)
            
            ### dashboard
            c1, c2 = st.columns(2)
            c1.metric(label='No Shows',
                       value=len(final_df[final_df['NO SHOW (Y/N)']=='YES']))
            c2.metric(label='Shows',
                       value=len(final_df[final_df['NO SHOW (Y/N)']!='YES']))

            ### download report button
            csv_string = final_df.to_csv(index=False)  # convert to predcitions df to csv
            export_report_button = st.download_button("Download Report",
                                                      csv_string,
                                                      file_name="final_report.csv",
                                                      mime="text/csv")
        else:
            st.error("Error in backend processing.")

    ### links
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("[🏥 CHLA](https://www.chla.org/)")
    st.markdown("[🐱 GitHub](https://github.com/svanhemert00/chla-no-show-web-app)")
    
if __name__=='__main__':
    main()