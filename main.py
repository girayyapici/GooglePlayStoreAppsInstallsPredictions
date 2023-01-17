
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
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np

#!pip install streamlit
#!pip install hyperopt

st.header("Google Play Store Apps Installs Predictions")
st.text_input("Enter your Name: ", key="name")
df_ = pd.read_csv("googleplaystore.csv")
df=df_.copy()

#encoder = LabelEncoder()
#encoder.classes_ = np.load('classes.npy', allow_pickle=True)

if st.checkbox('Show dataframe'):
    df

df.columns = df.columns.str.replace(' ', '_')

df.Size=df.Size.str.replace('k','e+3')
df.Size=df.Size.str.replace('M','e+6')

df.Size=df.Size.replace('Varies with device',np.nan)
df.Size=df.Size.replace('1,000+',1000)
df.Size=pd.to_numeric(df.Size)

df.Installs=df.Installs.apply(lambda x: x.strip('+'))
df.Installs=df.Installs.apply(lambda x: x.replace(',',''))
df.Installs=df.Installs.replace('Free',np.nan)
df.Installs.str.isnumeric().sum()
df.Installs=pd.to_numeric(df.Installs)

df.Reviews.str.isnumeric().sum()
df[~df.Reviews.str.isnumeric()]
df=df.drop(df.index[10472])
df.Reviews=pd.to_numeric(df.Reviews)

print("Range: ", df.Rating.min(), "-", df.Rating.max())
df.Rating.dtype
print(df.Rating.isna().sum(), "null values out of", len(df.Rating))

df.Price.unique()
df.Price = df.Price.apply(lambda x: x.strip('$'))
df.Price = pd.to_numeric(df.Price)

sep = ';'
rest = df.Genres.apply(lambda x: x.split(sep)[0])
df['Pri_Genres'] = rest
df.Pri_Genres.head()

rest = df.Genres.apply(lambda x: x.split(sep)[-1])
rest.unique()
df['Sec_Genres'] = rest
df.Sec_Genres.head()

df["Rating"] = df["Rating"].fillna(df.groupby("Pri_Genres")["Rating"].transform("mean"))
df["Size"] = df["Size"].fillna(df.groupby("Pri_Genres")["Size"].transform("mean"))

le = preprocessing.LabelEncoder()
df['App'] = le.fit_transform(df['App'])

le = preprocessing.LabelEncoder()
df['Pri_Genres'] = le.fit_transform(df['Pri_Genres'])

le = preprocessing.LabelEncoder()
df['Content_Rating'] = le.fit_transform(df['Content_Rating'])

#df['Type'] = pd.get_dummies(df['Type']) kontrol edilecek

le = preprocessing.LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# load model
#best_xgboost_model = xgb.XGBRegressor()
#best_xgboost_model.load_model("best_model.json")

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

