
# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디씨전트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수: bill_length_mm

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

penguins = load_penguins()
penguins
penguins.isna().sum()

train, test = train_test_split(penguins,test_size=0.2,random_state=42)

## 숫자형 채우기
quantitative_train = train.select_dtypes(include = [int, float])
quant_selected_train = quantitative_train.columns[quantitative_train.isna().sum() > 0]

quantitative_test = test.select_dtypes(include = [int, float])
quant_selected_test = quantitative_test.columns[quantitative_test.isna().sum() > 0]

for col in quant_selected_train:
    train[col].fillna(train[col].mean(), inplace=True)

for col in quant_selected_test:
    test[col].fillna(test[col].mean(), inplace=True)

## 범주형 채우기
qualitative_train = train.select_dtypes(include = [object])
qual_selected_train = qualitative_train.columns[qualitative_train.isna().sum() > 0]

qualitative_test = train.select_dtypes(include = [object])
qual_selected_test = qualitative_train.columns[qualitative_train.isna().sum() > 0]

for col in qual_selected_train:
    train[col].fillna("unknown", inplace=True)
    
for col in qual_selected_test:
    test[col].fillna("unknown", inplace=True)

train_n=len(train)

# 통합 df 만들기 + 더미코딩
df = pd.concat([train, test], ignore_index=True)

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

y_train = train_df['bill_length_mm']
y_test = test_df['bill_length_mm']
X_train = train_df.drop(columns='bill_length_mm')
X_test = train_df.drop(columns='bill_length_mm')

dct_model = DecisionTreeRegressor(random_state=42)
param_grid_dct={
    'max_depth': np.arange(7,20,1),
    'min_samples_split': np.arange(10,30,1)
}

np.random.seed(42)
grid_search=GridSearchCV(
    estimator=dct_model,
    param_grid=param_grid_dct,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(X_train, y_train)

grid_search.best_params_
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

y_pred_dct = best_model.predict(X_test) # predict 함수 사용가능

mse_dct = mean_squared_error(y_test, y_pred_elast)
r2_dct = r2_score(y_test, y_pred_dct)


elast_model= ElasticNet()

param_grid={
    'alpha': np.arange(0,100,1),
    'l1_ratio': np.arange(0.,1,0.1)
}

np.random.seed(42)
grid_search=GridSearchCV(
    estimator=elast_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(X_train, y_train)

grid_search.best_params_
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

y_pred_elast = best_model.predict(X_test) # predict 함수 사용가능

mse_elast = mean_squared_error(y_test, y_pred_elast)
r2_elast = r2_score(y_test, y_pred_elast)
