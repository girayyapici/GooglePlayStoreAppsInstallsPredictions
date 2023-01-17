
# App         : Uygulama Adı
# Category    : Uygulamanın ait olduğu kategori
# Rating      : Uygulamanın genel kullanıcı değerlendirmesi
# Reviews     : Yorum Sayısı
# Size        : Boyutu
# Installs    : İndirme Sayısı
# Type        : Ücretli veya Ücretsiz oluşu
# Price       : Fiyatı
# Content Rating  : Hedeflenen yaş grubu
# Genres      : Ana kategorisi dışında başka bir türede ait olabilir.
# Last Updated: Son güncellenme Tarihi
# Current Ver :
# Android Ver :

import streamlit as st
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np

#!pip install streamlit
#!pip install hyperopt

st.header("Google Play Store Apps Installs Predictions")
st.text_input("Enter your Name: ", key="name")
df_ = pd.read_csv("Proje/googleplaystore.csv")
df=df_.copy()

encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show dataframe'):
    df

input_Size = st.slider('Size', 0.0, max(df["Size"]), 1.0)
input_Type = st.slider('Type', 0.0, max(df["Type"]), 1.0)
input_Price = st.slider('Price', 0.0, max(df["Price"]), 1.0)
input_Content_Rating = st.slider('Content_Rating', 0.0, max(df["Content_Rating"]), 1.0)
input_Pri_Genres = st.slider('Pri_Genres', 0.0, max(df["Pri_Genres"]), 1.0)

if st.button('Make Prediction'):
    inputs = np.expand_dims(
       [input_Size, input_Type, input_Price, input_Content_Rating, input_Pri_Genres], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your app will get: {np.squeeze(prediction, -1)} downloads")
    st.write(f"Thank you {st.session_state.name}!")

