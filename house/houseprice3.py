# y = 2x+3의 그래프를 그려보세요!
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 직선의 모수는 기울기와 절편(기울기의 절댓값이 크면 y축에 달라붙고, y절편이 크면 직선이 위로 올라간다.)
a = 1
b = 3
x = np.linspace(-5,5,100)
y = a * x + b
plt.axvline(0,color='black', linewidth=0.5)
plt.axhline(0,color='black', linewidth=0.5)
plt.plot(x,y,color='blue')    
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()
plt.clf()

# a:기울기, b:y절편일때 y축이 집값일때 x축은 방의 갯수
house_train = pd.read_csv('house/train.csv')
my_df = house_train[['BedroomAbvGr','SalePrice']].head(10)
my_df['SalePrice'] = my_df['SalePrice'] / 1000 
plt.figure(figsize=(10, 6))
plt.scatter(x=my_df['BedroomAbvGr'],y=my_df['SalePrice'])
a=70
b=10
x= np.linspace(0,5,100)
y = a*x+b
plt.plot(x,y, color='blue')
plt.show()
plt.clf()

# 테스트 집 정보 가져오기
house_test = pd.read_csv('house/test.csv')
(a * house_test['BedroomAbvGr'] +b) * 1000


# sub 데이터 불러오기
house_sub = pd.read_csv('house/sample_submission.csv')
house_sub 

# SalePrice 바꿔치기
house_sub['SalePrice']= (a * house_test['BedroomAbvGr'] +b) * 1000
house_sub
house_sub.to_csv('house/sample_submission11.csv', index=False)

# 직선 성능 평가 1 : 절대값으로로
a=80
b=-30

# y_hat 어떻게 구할까?
y_hat = (a * house_train['BedroomAbvGr'] + b) * 1000 

# y는 어디에 있는가?
y = house_train['SalePrice']

# 절대거리 :abs
# 1조: a=70, b=10일때 직선의 성능: np.int64(106021410)/np.int64(13029269913338)
# 2조: a=12, b=170 : 94512422 / 9770646581338
# 3조: a=53, b=45 : 93754868 / 10623131755338
# 4조: a=36, b=68 : 81472836 / 9459298301338
# 5조: a=80, b=-30 : 103493158 / 13371156733338

np.abs(y - y_hat).sum()

# 직선 성능 평가 2 : 정사각형의 넓이의 합(제곱곱)

np.sum((y - y_hat)**2)

# !pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성 (절댓값으로 구하는 2번 지표)
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


# house에 적용
x = np.array(house_train['BedroomAbvGr']).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array(house_train['SalePrice']) / 1000
# y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성 (절댓값으로 구하는 2번 지표)
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)
y_pred

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

house_train['BedroomAbvGr'].value_counts()
house_test['BedroomAbvGr'].value_counts()

test_x = np.array(house_test['BedroomAbvGr']).reshape(-1,1)
test_x
pred_y = model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
house_sub['SalePrice'] = pred_y *1000
house_sub 

house_sub.to_csv('house/sample_submission12.csv', index=False)


#회귀직선 구하기

# y= x^2+3, my_f(x)의 최솟값:3 (x=0일떄)
def my_f(x):
    return x**2+3

def my_f2(x):
    return x[0]**2 + x[1]**2 + 3
my_f2([1,3])

def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 + 7
my_f3([1,2,3])

import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [10]
initial_guess2 = [-10,3]
initial_guess3 = [-10,3,4]

# 최솟값 찾기
result= minimize(my_f, initial_guess)
result2= minimize(my_f2, initial_guess2)
result3= minimize(my_f3, initial_guess3)

# 결과 출력
result.fun # 최소값
result.x #최소값을 갖는 x 값
result2.fun # 최소값
result2.x #최소값을 갖는 x 값
result3.fun # 최소값
result3.x #최소값을 갖는 x 값


# house에 적용2
x = np.array(house_train['TotalBsmtSF']).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array(house_train['SalePrice'])
# y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성 (절댓값으로 구하는 2번 지표)
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)
y_pred
import numpy as np

house_test['TotalBsmtSF'].isna().sum()
test_x = np.array(house_test['TotalBsmtSF']).reshape(-1,1)
test_x
pred_y = model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
house_sub['SalePrice'] = pred_y
house_sub 


house_sub.to_csv('house/sample_submission30.csv', index=False)


