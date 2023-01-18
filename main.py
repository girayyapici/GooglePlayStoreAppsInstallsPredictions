
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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np

#!pip install streamlit
#!pip install hyperopt

st.header("Google Play Store Apps Installs Predictions")
st.text_input("Enter your Name: ", key="name")
df_ = pd.read_csv("https://raw.githubusercontent.com/girayyapici/GooglePlayStoreAppsInstallsPredictions/main/googleplaystore.csv")
df=df_.copy()

encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy', allow_pickle=True)

if st.checkbox('Show dataframe'):
    df

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

input_Size = st.slider('Size', min(df["Size"]), max(df["Size"]))
input_Type = st.slider('Type', min(df["Type"]), max(df["Type"]))
input_Price = st.slider('Price', min(df["Price"]), max(df["Price"]))
input_Content_Rating = st.slider('Content_Rating', min(df["Content_Rating"]), max(df["Content_Rating"]))
input_Pri_Genres = st.slider('Pri_Genres', min(df["Pri_Genres"]), max(df["Pri_Genres"]))

if st.button('Make Prediction'):
    inputs = np.expand_dims(
       [input_Size, input_Type, input_Price, input_Content_Rating, input_Pri_Genres], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your app will get: {np.squeeze(prediction, -1)} downloads")
    st.write(f"Thank you {st.session_state.name}!")

