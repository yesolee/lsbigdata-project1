# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train = pd.read_csv('house/train.csv')
house_test = pd.read_csv('house/test.csv')
sub_df = pd.read_csv('house/sample_submission.csv')

## 회귀분석 적합(fit)하기
x = house_train.select_dtypes(include=[int,float])
x.info()
# 필요없는 칼럼 제거하기
x = x.iloc[:,1:-1]
# x.drop(columns = ['Id','SalePrice'])
y = house_train['SalePrice']

# 결측값 확인
x.isna().sum() 
fill_values = {
    'LotFrontage' : x['LotFrontage'].mean(),
    'MasVnrArea' : x['MasVnrArea'].mean(),
    'GarageYrBlt': x['GarageYrBlt'].mean() 
}
# mean, mode(최빈값값), quantile, 중앙값 등 중 뭐가 잘 나오는지 다 해보기

x = x.fillna(value=fill_values)
# LotFrontage, MasVnrArea, GarageYrBlt

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 예측값 계산
test_x = house_test.select_dtypes(include=[int,float])
test_x = x_test.iloc[:,1:]

#결측치 채우기
test_x = test_x.fillna(x_test.mean())
test_x.isna().sum() 

y_pred = model.predict(test_x)

# SalePrice 바꿔치기
sub_df['SalePrice'] = y_pred

sub_df.to_csv('house/sample_submission35.csv', index=False)

