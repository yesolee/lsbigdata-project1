# 57페이지 신뢰구간 구하기 연습문제 2)

x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
len(weights)
x.mean()
# 표준편차 6kg, 모평균에 대한 90% 신뢰구간 구하기

# X bar ~N(mu, sigma^2/n)
# X bar ~N(68.89375, 6^2)
from scipy.stats import norm

z_005 = norm.ppf(0.95, loc=0, scale=1)
z_005
x.mean() + z_005 * 6 / np.sqrt(16)
x.mean() - z_005 * 6 / np.sqrt(16)

# X ~ N(3,5^2) 데이터로부터 E[X^2]구하기
x = norm.rvs(loc=3, scale=5, size = 100000)
x2 = x**2
x2.mean()
sum(x**2) / (len(x) - 1)

# X ~ N(3,5^2) 데이터로부터 E[X^2]구하기
x = norm.rvs(loc=3, scale=5, size = 100000)
x2 = (x-x**2)/(2*x)
x2.mean()
sum(x2) / (len(x2) - 1)

# 몬테카를로 적분: 확률변수 기대값을 구할 떄, 표본을 많이 뽑은 후, 원하는 형태로 변형, 평균을 계산해서 기대값을 구하는 방법

np.random.seed(20240729)
x = norm.rvs(loc=3, scale=5, size = 100000)
x_bar  = x.mean()
s_2 = sum((x-x_bar)**2) / (100000-1)
s_2

#np.var(x) # n으로 나눈 값 (기본) 사용하면 안됨 주의!!
np.var(x, ddof=1) # n-1로 나눈 값 (표본분산)

# n-1 하는 이유! 불편추정량: n-1로해야 sigma^2가 나온다.
x=norm.rvs(loc=3, scale=5, size=20)
np.var(x)
np.var(x,ddof=1)

