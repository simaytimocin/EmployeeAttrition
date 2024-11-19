import pickle
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

model = pickle.load(open("xgb_model.pkl", 'rb'))
scaler = pickle.load(open("mm_scaler.pkl", 'rb'))
features = pickle.load(open("xgb_features.pkl", 'rb'))

def attrition():
    st.set_page_config(page_title="Çalışan Ayrılma Tahmini", page_icon="💼", layout="wide")
    st.title("💼 Çalışan Ayrılma Tahmin Uygulaması")
    st.markdown("""
    <style>
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/cubes.png');
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.header("Çalışan Bilgileri Formu")

    with st.sidebar.form("attrition_form"):
        st.markdown("**Lütfen aşağıdaki çalışan bilgilerini doldurun:**")
        
        age = st.number_input("Yaş", min_value=18, max_value=80, help="18-80")
        business_travel = st.radio("İş Seyahati", ["Nadiren", "Sıkça", "Seyahat Yok"])
        department = st.radio("Departman", ["Araştırma & Geliştirme", "İnsan Kaynakları", "Satış"])
        distance_from_home = st.number_input("Eve Uzaklık", min_value=1, max_value=29, help="1-29")
        education = st.radio("Eğitim Seviyesi", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Eğitim Alanı", [
            "Yaşam Bilimleri", "Tıp", "Pazarlama", "Teknik Derece", "İnsan Kaynakları", "Diğer"])
        environment_satisfaction = st.radio("Çevre Memnuniyeti", [1, 2, 3, 4])
        gender = st.radio("Cinsiyet", ["Erkek", "Kadın"])
        job_involvement = st.number_input("İşe Katılım", min_value=1, max_value=4, help="1-4")
        job_level = st.number_input("İş Seviyesi", min_value=1, max_value=5, help="1-5")
        job_role = st.selectbox("İş Rolü", [
            "Satış Temsilcisi", "Araştırma Bilimcisi", "Laboratuvar Teknisyeni", "Üretim Direktörü",
            "Sağlık Temsilcisi", "Yönetici", "Satış Temsilcisi", "Araştırma Direktörü",
            "İnsan Kaynakları"])
        job_satisfaction = st.radio("İş Memnuniyeti", [1, 2, 3, 4])
        marital_status = st.selectbox("Medeni Durum", ["Evli", "Bekar", "Boşanmış"])
        monthly_income = st.number_input("Aylık Gelir", min_value=1000, max_value=20000, help="1000-20000")
        num_companies_worked_in = st.number_input("Çalışılan Şirket Sayısı", min_value=0, max_value=9, help="0-9")
        overtime = st.radio("Fazla Mesai", ["Evet", "Hayır"])
        performance_rating = st.number_input("Performans Değerlendirmesi", min_value=1, max_value=4, help="1-4")
        relationship_satisfaction = st.number_input("İlişki Memnuniyeti", min_value=1, max_value=4, help="1-4")
        total_working_years = st.number_input("Toplam Çalışma Yılları", min_value=0, max_value=40, help="0-40")
        training_times_last_year = st.number_input("Geçen Yıl Eğitim Süresi", min_value=0, max_value=6, help="0-6")
        work_life_balance = st.number_input("İş-Yaşam Dengesi", min_value=1, max_value=4, help="1-4")
        years_at_company = st.number_input("Şirkette Geçirilen Yıl", min_value=0, max_value=40, help="0-40")
        years_in_current_role = st.number_input("Mevcut Rolde Geçen Yıl", min_value=0, max_value=18, help="0-18")
        years_since_last_promotion = st.number_input("Son Terfiden Beri Geçen Yıl", min_value=0, max_value=15, help="0-15")
        years_with_curr_manager = st.number_input("Mevcut Yönetici ile Geçen Yıl", min_value=0, max_value=17, help="0-17")

        submitted = st.form_submit_button("Ayrılma Tahmini Yap", help="Çalışanın ayrılma durumunu tahmin etmek için tıklayın")

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
            <h2>Tahmin Sonucu</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.success("Ayrılma Tahmini: Evet", icon="⚠️")
            st.markdown("**Model, çalışanın şirketten ayrılma ihtimalinin yüksek olduğunu tahmin ediyor.**")
        else:
            st.success("Ayrılma Tahmini: Hayır", icon="✅")
            st.markdown("**Model, çalışanın şirkette kalma ihtimalinin yüksek olduğunu tahmin ediyor.**")

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
    """, unsafe_allow_html=True)

attrition()
