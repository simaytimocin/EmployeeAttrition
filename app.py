import pickle
import numpy as np
import pandas as pd
import streamlit as st


model = pickle.load(open("xgb_model.pkl", 'rb'))
scaler = pickle.load(open("mm_scaler.pkl", 'rb'))
features = pickle.load(open("xgb_features.pkl", 'rb'))

def attrition():
    st.title("Employee Attrition Prediction App")

    with st.form("attrition_form"):

        st.header("Employee Details")

        
        age = st.number_input("Age", min_value=18, max_value=80, help="18-80")
        business_travel = st.radio("Business Travel", ["Rarely", "Frequently", "No Travel"])
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1600, help="100-1600")
        department = st.radio("Department", ["Research & Development", "Human Resources", "Sales"])
        distance_from_home = st.number_input("Distance From Home", min_value=1, max_value=29, help="1-29")
        education = st.radio("Education", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
        environment_satisfaction = st.radio("Environment Satisfaction", [1, 2, 3, 4])
        gender = st.radio("Gender", ["Male", "Female"])
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, help="30-100")
        job_involvement = st.number_input("Job Involvement", min_value=1, max_value=4, help="1-4")
        job_level = st.number_input("Job Level", min_value=1, max_value=5, help="1-5")
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
            "Healthcare Representative", "Manager", "Sales Representative", "Research Director",
            "Human Resources"])
        job_satisfaction = st.radio("Job Satisfaction", [1, 2, 3, 4])
        marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, help="1000-20000")
        num_companies_worked_in = st.number_input("No of Companies Worked in", min_value=0, max_value=9, help="0-9")
        overtime = st.radio("Over Time", ["Yes", "No"])
        performance_rating = st.number_input("Performance Rating", min_value=1, max_value=4, help="1-4")
        relationship_satisfaction = st.number_input("Relationship Satisfaction", min_value=1, max_value=4, help="1-4")
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, help="0-40")
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, help="0-6")
        work_life_balance = st.number_input("Work Life Balance", min_value=1, max_value=4, help="1-4")
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, help="0-40")
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=18, help="0-18")
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, help="0-15")
        years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=17, help="0-17")

        submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        
        input_data = {
            "Age": age,
            "BusinessTravel": business_travel,
            "DailyRate": daily_rate,
            "Department": department,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "NumCompaniesWorked": num_companies_worked_in,
            "OverTime": overtime,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times_last_year,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager,
        }

        df = pd.DataFrame([input_data])

        
        df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                                    df['JobInvolvement'] +
                                    df['JobSatisfaction'] +
                                    df['RelationshipSatisfaction'] +
                                    df['WorkLifeBalance']) / 5
        df.drop(['EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'RelationshipSatisfaction',
                 'WorkLifeBalance'], axis=1, inplace=True)

        
        df = pd.get_dummies(df)

        
        df = df.reindex(columns=features, fill_value=0)

        
        df_scaled = scaler.transform(df)

        
        prediction = model.predict(df_scaled)

        
        if prediction[0] == 1:
            st.write("Attrition Predicted: Yes")
        else:
            st.write("Attrition Predicted: No")


attrition()
