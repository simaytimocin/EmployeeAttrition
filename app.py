import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load pickle files
model = pickle.load(open("xgb_model.pkl", 'rb'))
scaler = pickle.load(open("mm_scaler.pkl", 'rb'))
features = pickle.load(open("xgb_features.pkl", 'rb'))

def attrition():
    st.title("Çalışan İşten Ayrılma Tahmin Uygulaması")

    # Create two columns: one for the form and one for the result
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Çalışan Bilgileri")

        # Input fields for user to provide employee details in Turkish
        age = st.number_input("Yaş", min_value=18, max_value=80, help="18-80")
        business_travel = st.radio("İş Seyahati", ["Nadiren", "Sıkça", "Seyahat Yok"])
        daily_rate = st.number_input("Günlük Ücret", min_value=100, max_value=1600, help="100-1600")
        department = st.radio("Departman", ["Araştırma ve Geliştirme", "İnsan Kaynakları", "Satış"])
        distance_from_home = st.number_input("Evden Uzaklık", min_value=1, max_value=29, help="1-29")
        education = st.radio("Eğitim Durumu", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Eğitim Alanı", [
            "Yaşam Bilimleri", "Tıp", "Pazarlama", "Teknik Alan", "İnsan Kaynakları", "Diğer"])
        environment_satisfaction = st.radio("Çevre Tatmini", [1, 2, 3, 4])
        gender = st.radio("Cinsiyet", ["Erkek", "Kadın"])
        hourly_rate = st.number_input("Saatlik Ücret", min_value=30, max_value=100, help="30-100")
        job_involvement = st.number_input("İş Katılımı", min_value=1, max_value=4, help="1-4")
        job_level = st.number_input("İş Seviyesi", min_value=1, max_value=5, help="1-5")
        job_role = st.selectbox("İş Pozisyonu", [
            "Satış Temsilcisi", "Araştırma Bilimcisi", "Laboratuvar Teknisyeni", "Üretim Direktörü",
            "Sağlık Temsilcisi", "Yönetici", "Satış Temsilcisi", "Araştırma Direktörü",
            "İnsan Kaynakları"])
        job_satisfaction = st.radio("İş Tatmini", [1, 2, 3, 4])
        marital_status = st.selectbox("Medeni Durum", ["Evli", "Bekar", "Boşanmış"])
        monthly_income = st.number_input("Aylık Gelir", min_value=1000, max_value=20000, help="1000-20000")
        num_companies_worked_in = st.number_input("Çalışılan Şirket Sayısı", min_value=0, max_value=9, help="0-9")
        overtime = st.radio("Mesai", ["Evet", "Hayır"])
        performance_rating = st.number_input("Performans Değerlendirmesi", min_value=1, max_value=4, help="1-4")
        relationship_satisfaction = st.number_input("İlişki Tatmini", min_value=1, max_value=4, help="1-4")
        stock_option_level = st.selectbox("Hisse Senedi Opsiyon Seviyesi", [0, 1, 2, 3])
        total_working_years = st.number_input("Toplam Çalışma Yılı", min_value=0, max_value=40, help="0-40")
        training_times_last_year = st.number_input("Geçen Yıldaki Eğitim Sayısı", min_value=0, max_value=6, help="0-6")
        work_life_balance = st.number_input("İş-Yaşam Dengesi", min_value=1, max_value=4, help="1-4")
        years_at_company = st.number_input("Şirketteki Yıl", min_value=0, max_value=40, help="0-40")
        years_in_current_role = st.number_input("Mevcut Pozisyondaki Yıl", min_value=0, max_value=18, help="0-18")
        years_since_last_promotion = st.number_input("Son Terfi Yılı", min_value=0, max_value=15, help="0-15")
        years_with_curr_manager = st.number_input("Mevcut Yönetici ile Yıl", min_value=0, max_value=17, help="0-17")

        submitted = st.form_submit_button("İşten Ayrılma Tahmin Et")

    with col2:
        if submitted:
            # Create DataFrame from input data
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

            # Feature engineering
            df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                                        df['JobInvolvement'] +
                                        df['JobSatisfaction'] +
                                        df['RelationshipSatisfaction'] +
                                        df['WorkLifeBalance']) / 5
            df.drop(['EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'RelationshipSatisfaction',
                     'WorkLifeBalance'], axis=1, inplace=True)

            # Convert categorical features to numerical using one-hot encoding
            df = pd.get_dummies(df)

            # Align input features with the model's expected features
            df = df.reindex(columns=features, fill_value=0)

            # Scale the input features
            df_scaled = scaler.transform(df)

            # Make predictions
            prediction = model.predict(df_scaled)

            # Display prediction result
            if prediction[0] == 1:
                st.write("Tahmin: Evet, Çalışan İşten Ayrılacak")
            else:
                st.write("Tahmin: Hayır, Çalışan İşten Ayrılmayacak")

# Run the app
attrition()
