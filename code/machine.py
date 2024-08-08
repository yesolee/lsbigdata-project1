import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 필요한 데이터 불러오기기
house_train = pd.read_csv('house/train.csv')
house_test = pd.read_csv('house/test.csv')
sub_df = pd.read_csv('house/sample_submission.csv')

#이상치 탐색
house_train['GrLivArea'].sort_values()
house_train.query("GrLivArea > 4500")

# 이상치 제거
house_train = house_train.query("GrLivArea <= 4500")

# 회귀분석 적합(fit)하기
x = house_train[['GrLivArea']]  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = house_train['SalePrice']
# 시리즈로 불러오면 reshape을 하기 위해 np.array를 한거고
# 판다스데이터는 2차원의 세로벡터로 봐서 따로 안해줘도 됌
# 1차원 벡터는 길이 length 2차원 벡터는 rows

# 선형 회귀 모델 생성 (절댓값으로 구하는 2번 지표)
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a,b
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")


def my_f(x):
    return x[0]*

# 예측값 계산
y_pred = model.predict(x)
y_pred

house_train['GrLivArea'].sort_values()

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0,5000])
plt.ylim([0,900000])
plt.legend()
plt.show()
plt.clf()

house_train['BedroomAbvGr'].value_counts()
house_test['GrLivArea'].value_counts()
house_test['GrLivArea'].describe()

test_x = np.array(house_test['GrLivArea']).reshape(-1,1)
test_x
pred_y = model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
house_sub['SalePrice'] = pred_y 
house_sub 

house_sub.to_csv('house/sample_submission31.csv', index=False)
