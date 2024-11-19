import pickle
import streamlit as st
import pandas as pd

def attrition():
    model = pickle.load(open('xgb_model.pkl', 'rb'))
    st.title("Çalışan Terk Tahmin Uygulaması")

    with st.form("attrition_form"):

        st.header("Çalışan Detayları")

        age = st.number_input("Yaş", min_value=18, max_value=80, help="18-80")
        business_travel = st.radio("İş Seyahati", ["Nadir", "Sık", "Seyahat Yok"])
        daily_rate = st.number_input("Günlük Ücret", min_value=100, max_value=1600, help="100-1600")
        department = st.radio("Departman", ["Araştırma ve Geliştirme", "İnsan Kaynakları", "Satış"])
        distance_from_home = st.number_input("Evden Uzaklık", min_value=1, max_value=29, help="1-29")
        education = st.radio("Eğitim", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Eğitim Alanı",
                                       ["Yaşam Bilimleri", "Tıp", "Pazarlama", "Teknik Derece", "İnsan Kaynakları",
                                        "Diğer"])
        environment_satisfaction = st.radio("Çevre Memnuniyeti", [1, 2, 3, 4])
        gender = st.radio("Cinsiyet", ["Erkek", "Kadın"])
        hourly_rate = st.number_input("Saatlik Ücret", min_value=30, max_value=100, help="30-100")
        job_involvement = st.number_input("İşe Katılım", min_value=1, max_value=4, help="1-4")
        job_level = st.number_input("İş Seviyesi", min_value=1, max_value=5, help="1-5")
        job_role = st.selectbox("İş Rolü",
                                ["Satış Yöneticisi", "Araştırma Bilimcisi", "Laboratuvar Teknisyeni",
                                 "Üretim Direktörü", "Sağlık Temsilcisi", "Yönetici", "Satış Temsilcisi",
                                 "Araştırma Direktörü", "İnsan Kaynakları"])
        job_satisfaction = st.radio("İş Memnuniyeti", [1, 2, 3, 4])
        marital_status = st.selectbox("Medeni Durum", ["Evli", "Bekar", "Boşanmış"])
        monthly_income = st.number_input("Aylık Gelir", min_value=1000, max_value=20000, help="1000-20000")
        num_companies_worked_in = st.number_input("Çalışılan Şirket Sayısı", min_value=0, max_value=9, help="0-9")
        overtime = st.radio("Fazla Mesai", ["Evet", "Hayır"])
        performance_rating = st.number_input("Performans Değerlendirmesi", min_value=1, max_value=4, help="1-4")
        relationship_satisfaction = st.number_input("İlişki Memnuniyeti", min_value=1, max_value=4, help="1-4")
        stock_option_level = st.selectbox("Hisse Senedi Seçeneği Seviyesi", [0, 1, 2, 3])
        total_working_years = st.number_input("Toplam Çalışma Yılları", min_value=0, max_value=40, help="0-40")
        training_times_last_year = st.number_input("Geçen Yılki Eğitim Sayısı", min_value=0, max_value=6, help="0-6")
        work_life_balance = st.number_input("İş-Yaşam Dengesi", min_value=1, max_value=4, help="1-4")
        years_at_company = st.number_input("Şirkette Geçen Yıllar", min_value=0, max_value=40, help="0-40")
        years_in_current_role = st.number_input("Mevcut Rolde Geçen Yıllar", min_value=0, max_value=18, help="0-18")
        years_since_last_promotion = st.number_input("Son Terfiden Bu Yana Geçen Yıllar", min_value=0, max_value=15,
                                                     help="0-15")
        years_with_curr_manager = st.number_input("Mevcut Yöneticiyle Geçen Yıllar", min_value=0, max_value=17, help="0-17")

        submitted = st.form_submit_button("Terk Tahmin Et")

    if submitted:
        # Veri sözlüğü oluşturma
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
        
        df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
        
        df['StockOptionLevel_bool'] = df['StockOptionLevel'].apply(lambda x: 1 if x > 1 else 0)
        
        df.drop(['DailyRate', 'HourlyRate', 'StockOptionLevel'], axis=1, inplace=True)

        df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Araştırma ve Geliştirme' else 0)
        df.drop('Department', axis=1, inplace=True)

        df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0)
        df.drop('DistanceFromHome', axis=1, inplace=True)

        df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratuvar Teknisyeni' else 0)
        df.drop('JobRole', axis=1, inplace=True)

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

        # Kategorik verileri sayısal veriye dönüştürme
        if business_travel == 'Nadir':
            df['BusinessTravel_Travel_Rarely'] = 1
            df['BusinessTravel_Travel_Frequently'] = 0
            df['BusinessTravel_Non-Travel'] = 0
        elif business_travel == 'Sık':
            df['BusinessTravel_Travel_Rarely'] = 0
            df['BusinessTravel_Travel_Frequently'] = 1
            df['BusinessTravel_Non-Travel'] = 0
        else:
            df['BusinessTravel_Travel_Rarely'] = 0
            df['BusinessTravel_Travel_Frequently'] = 0
            df['BusinessTravel_Non-Travel'] = 1
        df.drop('BusinessTravel', axis=1, inplace=True)

        if gender == 'Erkek':
            df['Gender_1'] = 1
            df['Gender_0'] = 0
        else:
            df['Gender_1'] = 0
            df['Gender_0'] = 1
        df.drop('Gender', axis=1, inplace=True)

        if marital_status == 'Evli':
            df['MaritalStatus_Married'] = 1
            df['MaritalStatus_Single'] = 0
            df['MaritalStatus_Divorced'] = 0
        elif marital_status == 'Bekar':
            df['MaritalStatus_Married'] = 0
            df['MaritalStatus_Single'] = 1
            df['MaritalStatus_Divorced'] = 0
        else:
            df['MaritalStatus_Married'] = 0
            df['MaritalStatus_Single'] = 0
            df['MaritalStatus_Divorced'] = 1
        df.drop('MaritalStatus', axis=1, inplace=True)

        if overtime == 'Evet':
            df['OverTime_0'] = 1
            df['OverTime_1'] = 0
        else:
            df['OverTime_0'] = 0
            df['OverTime_1'] = 1
        df.drop('OverTime', axis=1, inplace=True)

        # Tahmin yapma
        prediction = model.predict(df)

        # Tahmin sonucunu gösterme
        if prediction[0] == 1:
            st.write("Tahmin: Çalışan Terk Edecek")
        else:
            st.write("Tahmin: Çalışan Terk Etmeyecek")

attrition()
