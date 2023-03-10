
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
import time
import datetime
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluation_model(pred, y_val):
  score_MSE = round(mean_squared_error(pred, y_val),2)
  score_MAE = round(mean_absolute_error(pred, y_val),2)
  score_r2score = round(r2_score(pred, y_val),2)
  return score_MSE, score_MAE, score_r2score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_csv("googleplaystore1.csv")


#df.to_csv('googleplaystore1.csv')

features = ['Size', 'Type', 'Price', 'Content_Rating', 'Pri_Genres', 'App', 'Reviews', 'lastupdated']
X = df[features]
y = df['Installs']
####
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#label_encoder = LabelEncoder()
#x_train['Species'] = label_encoder.fit_transform(x_train['Species'].values)
#x_test['Species'] = label_encoder.transform(x_test['Species'].values)
#ave label encoder classes
#np.save('classes.npy', label_encoder.classes_)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")
pred = best_xgboost_model.predict(x_test)
score_MSE, score_MAE, score_r2score = evaluation_model(pred, y_test)
print(score_MSE, score_MAE, score_r2score)
###
loaded_encoder = LabelEncoder()
loaded_encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# print(x_test.shape)
# #input_species = loaded_encoder.transform(np.expand_dims("Parkki",0))
# #print(int(input_species))
# a = [input_Size, input_Type, input_Price, input_Content_Rating, input_Pri_Genres, input_App, input_Reviews, input_Last_Updated]
# inputs = np.expand_dims([int(a)],0)
# print(inputs.shape)
#prediction = best_xgboost_model.predict(inputs)
# print("final pred", np.squeeze(prediction,-1))