
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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np

from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluation_model(pred, y_val):
  score_MSE = round(mean_squared_error(pred, y_val), 2)
  score_MAE = round(mean_absolute_error(pred, y_val), 2)
  score_r2score = round(r2_score(pred, y_val), 2)
  return score_MSE, score_MAE, score_r2score


def models_score(model_name, train_data, y_train, val_data, y_val):
    model_list = ["Decision_Tree", "Random_Forest", "XGboost_Regressor"]
    # model_1
    if model_name == "Decision_Tree":
        reg = DecisionTreeRegressor(random_state=42)
    # model_2
    elif model_name == "Random_Forest":
        reg = RandomForestRegressor(random_state=42)

    # model_3
    elif model_name == "XGboost_Regressor":
        reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, )
    else:
        print("please enter correct regressor name")

    if model_name in model_list:
        reg.fit(train_data, y_train)
        pred = reg.predict(val_data)

        score_MSE, score_MAE, score_r2score = evaluation_model(pred, y_val)
        return round(score_MSE, 2), round(score_MAE, 2), round(score_r2score, 2)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("Proje/googleplaystore.csv")
df=df_.copy()

####
# Data Cleaning
####
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

print("Range: ", df.Rating.min(),"-",df.Rating.max())
df.Rating.dtype
print(df.Rating.isna().sum(),"null values out of", len(df.Rating))

df.Price.unique()
df.Price=df.Price.apply(lambda x: x.strip('$'))
df.Price=pd.to_numeric(df.Price)

sep = ';'
rest = df.Genres.apply(lambda x: x.split(sep)[0])
df['Pri_Genres']=rest
df.Pri_Genres.head()

rest = df.Genres.apply(lambda x: x.split(sep)[-1])
rest.unique()
df['Sec_Genres']=rest
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

df["Installs_qcut"] = pd.cut(df["Installs"], [0, 10000, 1000000, 5000000, 1000000000], labels=[1, 2, 3, 4])
le = preprocessing.LabelEncoder()
df['Installs_qcut'] = le.fit_transform(df['Installs_qcut'])
df.isnull().sum()
df.describe().T

features = ['App', 'Reviews', 'Size', 'Rating', 'Type', 'Price', 'Content_Rating', 'Pri_Genres']
X = df[features]
y = df['Installs']
####
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#label_encoder = LabelEncoder()
#x_train['Species'] = label_encoder.fit_transform(x_train['Species'].values)
#x_test['Species'] = label_encoder.transform(x_test['Species'].values)
model_list = ["Decision_Tree", "Random_Forest", "XGboost_Regressor"]
####
result_scores = []
for model in model_list:
    score = models_score(model, x_train, y_train, x_test, y_test)
    result_scores.append((model, score[0], score[1],score[2]))
    print(model, score)

df_result_scores = pd.DataFrame(result_scores, columns=["model", "mse", "mae", "r2score"])
df_result_scores
####
num_estimator = [100, 150, 200, 250]

space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
         'gamma': hp.uniform('gamma', 1, 9),
         'reg_alpha': hp.quniform('reg_alpha', 30, 180, 1),
         'reg_lambda': hp.uniform('reg_lambda', 0, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
         'n_estimators': hp.choice("n_estimators", num_estimator),
         }


def hyperparameter_tuning(space):
    model = xgb.XGBRegressor(n_estimators=space['n_estimators'], max_depth=int(space['max_depth']),
                             gamma=space['gamma'],
                             reg_alpha=int(space['reg_alpha']), min_child_weight=space['min_child_weight'],
                             colsample_bytree=space['colsample_bytree'], objective="reg:squarederror")

    score_cv = cross_val_score(model, x_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
    return {'loss': -score_cv, 'status': STATUS_OK, 'model': model}


trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print(best)
####
best['max_depth'] = int(best['max_depth']) # convert to int
best["n_estimators"] = num_estimator[best["n_estimators"]] # assing n_estimator because it returs the index
best_xgboost_model = xgb.XGBRegressor(**best)
best_xgboost_model.fit(x_train, y_train)
pred = best_xgboost_model.predict(x_test)
score_MSE, score_MAE, score_r2score = evaluation_model(pred, y_test)
to_append = ["XGboost_hyper_tuned",score_MSE, score_MAE, score_r2score]
df_result_scores.loc[len(df_result_scores)] = to_append

best_xgboost_model.save_model("best_model.json")
####