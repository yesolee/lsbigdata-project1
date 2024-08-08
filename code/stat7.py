# 그래프 그리기 y=2*x+3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# x의 값의 범위 설정
x=np.linspace(0,100,400)

# y값 계산
y= 2*x +3
plt.plot(x,y,color='black')
plt.scatter(obs_x, obs_y, color='blue', s=1)
# 
# np.random.seed(20240805)
obs_x= np.random.choice(np.arange(100),20)
epsilon_i = norm.rvs(loc=0, scale=10, size=20)
obs_y= 2*obs_x + 3 + epsilon_i

import pandas as pd

df = pd.DataFrame({
    "x": obs_x,
    "y": obs_y
})
df

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# 모델 학습
obs_x = obs_x.reshape(-1,1)
model.fit(obs_x,obs_y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a hat
model.intercept_ # 절편 b hat

# 그래프 그리기
x = np.linspace(0,100,400)
y = model.coef_[0] * x + model.intercept_
plt.plot(x,y,color='red') # 회귀직선
plt.xlim([0,100])
plt.ylim([0,300])
plt.show()
plt.clf()


#!pip install statsmodels
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())





import numpy as np
8.79/np.sqrt(20)


from scipy.stats import norm
from scipy.stats import norm

1-norm.cdf(18,loc=10,scale=1.96)

# 교재 57페이지
# 표본에 의해 판단해볼때, 1등급으로 판단 가능?
# 1등급 기준은 평균 복합 어네지 소비효율이 16.0 이상인 경우 부여
# 신형 자동차 15대의 복합 에너지 소비효율 측정 결과

"가설: 1등급을 받을 것이다."
x= np.array([15.078,15.752,15.549,15.56,16.098,13.277,15.462,16.116,15.214,16.93,14.118,14.927,
 15.382, 16.709, 16.804])
x_bar = x.mean()
x_bar
s= x.std()
s
z = (x_bar - 16) / (s/np.sqrt(15)
z
p_value = 1- norm.cdf(z, loc=0, scale=1)
p_value

유의수준 = 0.01

# 유의수준 1%로 설정정
"기각(=1등급 못받는다)" if p_value < 유의수준 else "귀무가설이 맞다(= 1등급 받을 수 있다.)"

# 95% 신뢰구간
a = x_bar - (s/np.sqrt(15)) * norm.ppf(0.975, loc=0, scale=1)
b = x_bar + (s/np.sqrt(15)) * norm.ppf(0.975, loc=0, scale=1)
w
