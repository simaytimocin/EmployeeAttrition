import pickle
import streamlit as st
import pandas as pd


def attrition():
    model = pickle.load(open('xgb_model.pkl', 'rb'))
    st.title("Employee Attrition Prediction App")

    with st.form("attrition_form"):

        st.header("Employee Details")

        age = st.number_input("Age", min_value=18, max_value=80, help="18-80")
        business_travel = st.radio("Business Travel", ["Rarely", "Frequently", "No Travel"])
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1600, help="100-1600")
        department = st.radio("Department", ["Research & Development", "Human Resources", "Sales"])
        distance_from_home = st.number_input("Distance From Home", min_value=1, max_value=29, help="1-29")
        education = st.radio("Education", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Education Field",
                                       ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources",
                                        "Other"])
        environment_satisfaction = st.radio("Environment Satisfaction", [1, 2, 3, 4])
        gender = st.radio("Gender", ["Male", "Female"])
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, help="30-100")
        job_involvement = st.number_input("Job Involvement", min_value=1, max_value=4, help="1-4")
        job_level = st.number_input("Job level", min_value=1, max_value=5, help="1-5")
        job_role = st.selectbox("Job Role",
                                ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                 "Manufacturing Director",
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
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15,
                                                     help="0-15")
        years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=17, help="0-17")

        submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        # Data dictionary to store input values
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

        df.drop(
            ['EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'RelationshipSatisfaction',
             'WorkLifeBalance'],
            axis=1, inplace=True)

        df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x: 1 if x >= 2.8 else 0)
        df.drop('Total_Satisfaction', axis=1, inplace=True)

        df['Age_bool'] = df['Age'].apply(lambda x: 1 if x < 35 else 0)
        df.drop('Age', axis=1, inplace=True)

        df['DailyRate_bool'] = df['DailyRate'].apply(lambda x: 1 if x < 800 else 0)
        df.drop('DailyRate', axis=1, inplace=True)

        df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Research & Development' else 0)
        df.drop('Department', axis=1, inplace=True)

        df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0)
        df.drop('DistanceFromHome', axis=1, inplace=True)

        df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratory Technician' else 0)
        df.drop('JobRole', axis=1, inplace=True)

        df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
        df.drop('HourlyRate', axis=1, inplace=True)

        df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x: 1 if x < 4000 else 0)
        df.drop('MonthlyIncome', axis=1, inplace=True)

        df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x: 1 if x > 3 else 0)
        df.drop('NumCompaniesWorked', axis=1, inplace=True)

        df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x: 1 if x < 8 else 0)
        df.drop('TotalWorkingYears', axis=1, inplace=True)

        df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x: 1 if x < 3 else 0)
        df.drop('YearsAtCompany', axis=1, inplace=True)

        df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x: 1 if x < 3 else 0)
        df.drop('YearsInCurrentRole', axis=1, inplace=True)

        df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x < 1 else 0)
        df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

        df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x: 1 if x < 1 else 0)
        df.drop('YearsWithCurrManager', axis=1, inplace=True)

        # Convert Categorical to Numerical
        if business_travel == 'Rarely':
            df['BusinessTravel_Travel_Rarely'] = 1
            df['BusinessTravel_Travel_Frequently'] = 0
            df['BusinessTravel_Non-Travel'] = 0
        elif business_travel == 'Frequently':
            df['BusinessTravel_Travel_Rarely'] = 0
            df['BusinessTravel_Travel_Frequently'] = 1
            df['BusinessTravel_Non-Travel'] = 0
        else:
            df['BusinessTravel_Travel_Rarely'] = 0
            df['BusinessTravel_Travel_Frequently'] = 0
            df['BusinessTravel_Non-Travel'] = 1
        df.drop('BusinessTravel', axis=1, inplace=True)

        if education == 1:
            df['Education_1'] = 1
            df['Education_2'] = 0
            df['Education_3'] = 0
            df['Education_4'] = 0
            df['Education_5'] = 0
        elif education == 2:
            df['Education_1'] = 0
            df['Education_2'] = 1
            df['Education_3'] = 0
            df['Education_4'] = 0
            df['Education_5'] = 0
        elif education == 3:
            df['Education_1'] = 0
            df['Education_2'] = 0
            df['Education_3'] = 1
            df['Education_4'] = 0
            df['Education_5'] = 0
        elif education == 4:
            df['Education_1'] = 0
            df['Education_2'] = 0
            df['Education_3'] = 0
            df['Education_4'] = 1
            df['Education_5'] = 0
        else:
            df['Education_1'] = 0
            df['Education_2'] = 0
            df['Education_3'] = 0
            df['Education_4'] = 0
            df['Education_5'] = 1
        df.drop('Education', axis=1, inplace=True)

        if education_field == 'Life Sciences':
            df['EducationField_Life Sciences'] = 1
            df['EducationField_Medical'] = 0
            df['EducationField_Marketing'] = 0
            df['EducationField_Technical Degree'] = 0
            df['EducationField_Human Resources'] = 0
            df['EducationField_Other'] = 0
        elif education_field == 'Medical':
            df['EducationField_Life Sciences'] = 0
            df['EducationField_Medical'] = 1
            df['EducationField_Marketing'] = 0
            df['EducationField_Technical Degree'] = 0
            df['EducationField_Human Resources'] = 0
            df['EducationField_Other'] = 0
        elif education_field == 'Marketing':
            df['EducationField_Life Sciences'] = 0
            df['EducationField_Medical'] = 0
            df['EducationField_Marketing'] = 1
            df['EducationField_Technical Degree'] = 0
            df['EducationField_Human Resources'] = 0
            df['EducationField_Other'] = 0
        elif education_field == 'Technical Degree':
            df['EducationField_Life Sciences'] = 0
            df['EducationField_Medical'] = 0
            df['EducationField_Marketing'] = 0
            df['EducationField_Technical Degree'] = 1
            df['EducationField_Human Resources'] = 0
            df['Education_Other'] = 0
        elif education_field == 'Human Resources':
            df['EducationField_Life Sciences'] = 0
            df['EducationField_Medical'] = 0
            df['EducationField_Marketing'] = 0
            df['EducationField_Technical Degree'] = 0
            df['EducationField_Human Resources'] = 1
            df['EducationField_Other'] = 0
        else:
            df['EducationField_Life Sciences'] = 0
            df['EducationField_Medical'] = 0
            df['EducationField_Marketing'] = 0
            df['EducationField_Technical Degree'] = 0
            df['EducationField_Human Resources'] = 1
            df['EducationField_Other'] = 1
        df.drop('EducationField', axis=1, inplace=True)

        if gender == 'Male':
            df['Gender_1'] = 1
            df['Gender_0'] = 0
        else:
            df['Gender_1'] = 0
            df['Gender_0'] = 1
        df.drop('Gender', axis=1, inplace=True)

        if marital_status == 'Married':
            df['MaritalStatus_Married'] = 1
            df['MaritalStatus_Single'] = 0
            df['MaritalStatus_Divorced'] = 0
        elif marital_status == 'Single':
            df['MaritalStatus_Married'] = 0
            df['MaritalStatus_Single'] = 1
            df['MaritalStatus_Divorced'] = 0
        else:
            df['MaritalStatus_Married'] = 0
            df['MaritalStatus_Single'] = 0
            df['MaritalStatus_Divorced'] = 1
        df.drop('MaritalStatus', axis=1, inplace=True)

        if overtime == 'Yes':
            df['OverTime_0'] = 1
            df['OverTime_1'] = 0
        else:
            df['OverTime_0'] = 0
            df['OverTime_1'] = 1
        df.drop('OverTime', axis=1, inplace=True)

        if stock_option_level == 0:
            df['StockOptionLevel_0'] = 1
            df['StockOptionLevel_1'] = 0
            df['StockOptionLevel_2'] = 0
            df['StockOptionLevel_3'] = 0
        elif stock_option_level == 1:
            df['StockOptionLevel_0'] = 0
            df['StockOptionLevel_1'] = 1
            df['StockOptionLevel_2'] = 0
            df['StockOptionLevel_3'] = 0
        elif stock_option_level == 2:
            df['StockOptionLevel_0'] = 0
            df['StockOptionLevel_1'] = 0
            df['StockOptionLevel_2'] = 1
            df['StockOptionLevel_3'] = 0
        else:
            df['StockOptionLevel_0'] = 0
            df['StockOptionLevel_1'] = 0
            df['StockOptionLevel_2'] = 0
            df['StockOptionLevel_3'] = 1
        df.drop('StockOptionLevel', axis=1, inplace=True)

        if training_times_last_year == 0:
            df['TrainingTimesLastYear_0'] = 1
            df['TrainingTimesLastYear_1'] = 0
            df['TrainingTimesLastYear_2'] = 0
            df['TrainingTimesLastYear_3'] = 0
            df['TrainingTimesLastYear_4'] = 0
            df['TrainingTimesLastYear_5'] = 0
            df['TrainingTimesLastYear_6'] = 0
        elif training_times_last_year == 1:
            df['TrainingTimesLastYear_0'] = 0
            df['TrainingTimesLastYear_1'] = 1
            df['TrainingTimesLastYear_2'] = 0
            df['TrainingTimesLastYear_3'] = 0
            df['TrainingTimesLastYear_4'] = 0
            df['TrainingTimesLastYear_5'] = 0
            df['TrainingTimesLastYear_6'] = 0
        elif training_times_last_year == 2:
            df['TrainingTimesLastYear_0'] = 0
            df['TrainingTimesLastYear_1'] = 0
            df['TrainingTimesLastYear_2'] = 1
            df['TrainingTimesLastYear_3'] = 0
            df['TrainingTimesLastYear_4'] = 0
            df['TrainingTimesLastYear_5'] = 0
            df['TrainingTimesLastYear_6'] = 0
        elif training_times_last_year == 3:
            df['TrainingTimesLastYear_0'] = 0
            df['TrainingTimesLastYear_1'] = 0
            df['TrainingTimesLastYear_2'] = 0
            df['TrainingTimesLastYear_3'] = 1
            df['TrainingTimesLastYear_4'] = 0
            df['TrainingTimesLastYear_5'] = 0
            df['TrainingTimesLastYear_6'] = 0
        elif training_times_last_year == 4:
            df['TrainingTimesLastYear_0'] = 0
            df['TrainingTimesLastYear_1'] = 0
            df['TrainingTimesLastYear_2'] = 0
            df['TrainingTimesLastYear_3'] = 0
            df['TrainingTimesLastYear_4'] = 1
            df['TrainingTimesLastYear_5'] = 0
            df['TrainingTimesLastYear_6'] = 0
        elif training_times_last_year == 5:
            df['TrainingTimesLastYear_0'] = 0
            df['TrainingTimesLastYear_1'] = 0
            df['TrainingTimesLastYear_2'] = 0
            df['TrainingTimesLastYear_3'] = 0
            df['TrainingTimesLastYear_4'] = 0
            df['TrainingTimesLastYear_5'] = 1
            df['TrainingTimesLastYear_6'] = 0
        else:
            df['TrainingTimesLastYear_0'] = 0
            df['TrainingTimesLastYear_1'] = 0
            df['TrainingTimesLastYear_2'] = 0
            df['TrainingTimesLastYear_3'] = 0
            df['TrainingTimesLastYear_4'] = 0
            df['TrainingTimesLastYear_5'] = 0
            df['TrainingTimesLastYear_6'] = 1
        df.drop('TrainingTimesLastYear', axis=1, inplace=True)

        # Make predictions
        prediction = model.predict(df)

        st.write(prediction)

        # Display prediction result
        if prediction[0] == 1:
            st.write("Attrition Predicted: Yes")
        else:
            st.write("Attrition Predicted: No")


attrition()
