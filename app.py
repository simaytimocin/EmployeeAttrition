# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_axaSW95Whrdq3Ia44S2YG6WX579U4wW
"""




import pickle
import numpy as np
import streamlit as st

# Pickle dosyalarını yükle
model = pickle.load(open("xgb_model.pkl", 'rb'))
scaler = pickle.load(open("mm_scaler.pkl", 'rb'))
features = pickle.load(open("xgb_features.pkl", 'rb'))

# Kullanıcı arayüzü için Streamlit kullan
st.title("Çalışan Ayrılma Tahmin Uygulaması")
st.write("Bu uygulama, bir çalışanın şirkette kalıp kalmayacağını tahmin eder.")

# Kullanıcıdan girdi al
input_data = []
for feature in features:
    value = st.number_input(f"{feature} için değeri girin:", min_value=0.0, max_value=100.0, step=0.1)
    input_data.append(value)

# Girdi verilerini uygun hale getir
input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Tahmini yap ve sonucu göster
if st.button("Tahmin Et"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.write("Çalışanın ayrılması muhtemel.")
    else:
        st.write("Çalışan muhtemelen şirkette kalacak.")
