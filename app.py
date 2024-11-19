import pickle
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

model = pickle.load(open("xgb_model.pkl", 'rb'))
scaler = pickle.load(open("mm_scaler.pkl", 'rb'))
features = pickle.load(open("xgb_features.pkl", 'rb'))

def attrition():
    st.set_page_config(page_title="Employee Attrition Prediction", page_icon="💼", layout="wide")
    st.title("💼 Employee Attrition Prediction App")
    st.markdown("""
    <style>
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/cubes.png');
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.header("Employee Details Form")

    with st.sidebar.form("attrition_form"):
        st.markdown("**Please fill in the employee details below:**")
        
        age = st.number_input("Age", min_value=18, max_value=80, help="18-80")
        business_travel = st.radio("Business Travel", ["Rarely", "Frequently", "No Travel"])
        department = st.radio("Department", ["Research & Development", "Human Resources", "Sales"])
        distance_from_home = st.number_input("Distance From Home", min_value=1, max_value=29, help="1-29")
        education = st.radio("Education", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
        environment_satisfaction = st.radio("Environment Satisfaction", [1, 2, 3, 4])
        gender = st.radio("Gender", ["Male", "Female"])
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
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, help="0-40")
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, help="0-6")
        work_life_balance = st.number_input("Work Life Balance", min_value=1, max_value=4, help="1-4")
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, help="0-40")
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=18, help="0-18")
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, help="0-15")
        years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=17, help="0-17")

        submitted = st.form_submit_button("Predict Attrition", help="Click to predict employee attrition")

    if submitted:
        input_data = {
            "Age": age,
            "BusinessTravel": business_travel,
            "Department": department,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender,
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

        st.markdown("""
        <div style="text-align: center;">
            <h2>Prediction Result</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.success("Attrition Predicted: Yes", icon="⚠️")
            st.markdown("**The model predicts that the employee is likely to leave the company.**")
        else:
            st.success("Attrition Predicted: No", icon="✅")
            st.markdown("**The model predicts that the employee is likely to stay with the company.**")

    components.html("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """)

attrition()
