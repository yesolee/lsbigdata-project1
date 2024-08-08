# 원하는 변수를 사용해서 회귀모델을 만들고, 제출할것!
# 원하는 변수 2개
# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 라쏘, 릿지 회귀분석 => 범주형 더미코딩해서 하면 변수가 너무많아... 뭐슨 칼럼이 최적일까?찾는 모델

## 필요한 데이터 불러오기
house_train=pd.read_csv("house/train.csv")
house_test=pd.read_csv("house/test.csv")
sub_df=pd.read_csv("house/sample_submission.csv")

## 이상치 탐색
# house_train=house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
house_train["Neighborhood"].unique()

# 더미코드 
#### Neighborhood외의 열도 다 나옴
len(pd.get_dummies(
    house_train, columns=["Neighborhood"], drop_first=True).columns)

# Neighborhood만 나옴
Neighborhood_dummies = pd.get_dummies(
    house_train["Neighborhood"], drop_first=True)
len(Neighborhood_dummies.columns)
Neighborhood_dummies

x = house_train[["GrLivArea", "GarageArea"]]
x = pd.concat([house_train[["GrLivArea", "GarageArea"]] ,Neighborhood_dummies], axis=1)
x
y= house_train["SalePrice"]
y

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# test 데이터도 더미만들기
neighborhood_dummies_test = pd.get_dummies(
    house_test["Neighborhood"], drop_first=True)

test_x = pd.concat([house_test[["GrLivArea", "GarageArea"]] ,neighborhood_dummies_test], axis=1)
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("house/sample_submission51.csv", index=False)
