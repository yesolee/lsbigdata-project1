# !pip install scipy
from scipy.stats import bernoulli
#확률 질량 함수 pmf(k,p) : 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# P(x=1)
bernoulli.pmf(1, 0.3)
# P(x=0)
bernoulli.pmf(0, 0.3)

import numpy as np
from scipy.stats import binom
# 이항분포 X ~ P(X = k | n ,p)
# n: 베르누이 확률변수 더한 갯수
# p: 1이 나올 확률
# bino.pmf(k,n,p)
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

sum([ binom.pmf(i, n=30, p=0.3) for i in np.arange(31)])

# X ~ B(n, p)    
import math
math.factorial(54) / (math.factorial(26) * math.factorial(54-26))
math.comb(54,26)

# log(a*b) = log(a)+log(b)
np.cumprod(math.log(np.arange(1,55)))
math.log(24)
# math.log(1*2*3*4)= math.log(1)+math.log(2)+math.log(3)+math.log(4)
1*2*3*4
sum(np.log(np.arange(1,5)))
math.log(math.factorial(54))
logf_54 = sum(np.log(np.arange(1,55)))
logf_26 = sum(np.log(np.arange(1,27)))
logf_28 = sum(np.log(np.arange(1,29)))

logf_54 - (logf_26 +logf_28)
np.exp(logf_54 - (logf_26 +logf_28))

math.comb(2,1) * 0.3**1 * (1-0.3)**1

# X ~ B(n=10, p=0.36)
# p(X = 4) = ?
binom.pmf(4,10,0.36)
 
# p(x<=4) 0과 1이니까 0부터 10까지 나올 수 있음
np.arange(5) # 0 1 2 3 4
binom.pmf(np.arange(5), n=10, p=0.36).sum()

# p(2<x<=8)?
binom.pmf(np.arange(3,9), n=10, p=0.36).sum()

# X ~ B(30,0.2)
# p(x <4 or x>=25)
k = np.concat([np.arange(4),np.arange(25,31)])
binom.pmf(k,n=30,p=0.2).sum()

binom.pmf(np.arange(4),n=30,p=0.2).sum() + binom.pmf(np.arange(25,31),n=30,p=0.2).sum()

1-binom.pmf(np.arange(4,25),n=30,p=0.2).sum()

binom.pmf(np.arange(31),n=30,p=0.2).sum()

# rvs 함수(random variates sample)
# 표본 추출 함수
# X1 ~ Bernulli(p=0.3)
bernoulli.rvs(0.3) # 0또는 1이 나옴
# X2 ~ Bernulli(p=0.3)

# X~ B(n=2, p=0.3)  0,1,2
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(n=2, p=0.3, size=2) # 한번에 2개 뽑고 싶을 때


# x ~ B(30,0.26)
# 표본 30개를 뽑아보세요!
binom.rvs(n=30,p=0.26,size=30)

# X ~ B(30,0.26)
np.arang(31) * binom.rvs(n=30,p=0.26,size=30)
30 * 0.26
import matplotlib.pyplot as plt
import seaborn as sns
x = np.arange(31)
prob_x = binom.pmf(np.arange(31),30,0.26)
prob_x
sum(prob_x)
sns.barplot(prob_x)

import pandas as pd
df= pd.DataFrame({
    "x" :x,
    "prob": prob_x
})
# plt.figure(figsize=(10, 6))
sns.barplot(data = df, x='x', y = 'prob')
plt.show()
plt.clf()

# cdf(cumulative dist. function)
# 누적확률분포 함수
# F_X(x) P(X <= x)

binom.cdf(4, n=30, p=0.26)

binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26) 


binom.cdf(19, n=30, p=0.26) - binom.cdf(14, n=30, p=0.26) 

import numpy as np
import seaborn as sns


x_1 = binom.rvs(n=30, p=0.26, size=10)
x_1
    
x=np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x,color="blue")
# Add a point at (2,0)
plt.scatter(x_1, np.repeat(0.002,10),color='red', zorder = 100, s = 5 )
# 기댓값 표현
plt.axvline(x=7.8, color='green', linestyle='--', linewidth=2)

plt.show()
plt.clf()


# pmf : P(x=k)
# cdf : P(x<=k)
# rvs : random sample size (표본을 구해줌)

# 이항 분포 함수: 

# X ~ B(n,p)
# 앞면(1)이 나올 확률이 p인 동전을 n번 던져서 나온 앞면의 수
# (1번 나오면 베르누이가 됨)

# ppf : 
cdf p(x<=k)
binom.ppf(0.5, n=30, p=0.26)
binom.pmf(np.arange(9),n=30,p=0.26).sum()
binom.cdf(8, n=30, p=0.26)
binom.cdf(7, n=30, p=0.26)

binom.ppf(0.7, n=30, p=0.26)
binom.cdf(9, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)

1/np.sqrt(2*math.pi)

from scipy.stats import norm

norm.pdf(0, loc=0, scale=1)
norm.pdf(5, loc=3, scale=4)

#정규분포 pdf 그리기
# x= -3 ~ 3까지의 pdf 값 5개
k= np.linspace(-5,5,100)
k
y= norm.pdf(k, loc=0, scale=1)
y
plt.plot(k,y,color='black')
plt.show()
plt.clf()
ㄴ
# 모수(파라미터): 특징을 결정하는 수. 이항분포에서는 n,p 정규분포는 mu,sigma
# 베르누이는 모수가 1개(p)

## 평균 mu (loc): 분포의 중심 결정하는 모수
k= np.linspace(-5,5,100)
k
y= norm.pdf(k, loc=0, scale=1)
y
plt.plot(k,y,color='black')
plt.show()
plt.clf()

## 표준편차 sigma (scale): 분포의 퍼짐 결정하는 모수
k= np.linspace(-5,5,100)
k
y= norm.pdf(k, loc=0, scale=1)
y
y2= norm.pdf(k, loc=0, scale=2)
y3= norm.pdf(k, loc=0, scale=0.5)
plt.plot(k,y,color='black')
plt.plot(k,y2,color='red')
plt.plot(k,y3,color='blue')

plt.show()
plt.clf()

norm.cdf(0,loc=0,scale=1)
norm.cdf(100,loc=0,scale=1)

# mu=0,sigma=1 P(-2<X<0.54)?
norm.cdf(0.54,loc=0,scale=1) - norm.cdf(-2,loc=0,scale=1)

# P(x<1 or x>3)
norm.cdf(1,0,1) + 1 - norm.cdf(3,0,1)

# X ~ N(3,5^2)일때 P(3<X<5)? 15.54%
norm.cdf(5,3,5) - norm.cdf(3,3,5)
 
#  X ~ N(3,5^2)에서 표본 100개 뽑아보자!
x = norm.rvs(loc=3, scale=5, size = 1000)
x
sum((x>3) & (x<5))

# 평균0, 표준편차1
# 표본 1000개 뽑아서 0보다 작은 비율 확인: 50% 나오는가?
x = norm.rvs(loc=0, scale=1, size = 1000)
(x<0).mean()




x=norm.rvs(loc=3,scale=2,size=1000)
x
sns.histplot(x,stat="density")
plt.show()

# histoplot의 y축은 count, pdf는 y의 값이고, 면적이 확률인거지 pdf(높이)가 확률은 아님.


# plot the normal distribution PDF
xmin, xmax= (x.min(), x.max())
x_values=np.linspace(xmin,xmax,100)
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()


# 숙제 Qmd
# 1. 정규분포 pdf값을 계산하는 자신만의 파이썬 함수를 정의하고,
# 정규분포 mu=3, sigma=2의 pdf를 그릴 것
# X ~ N(3,2)

def npdf(x, mu,sigma):
    import math
    part1 = 1 / (sigma * math.sqrt(2 * math.pi))
    part2 = math.exp((-1/2) * ((x-mu)/math.pi)**2)
    y = part1 * part2
    return y

x_values = np.linspace(-3*2,3*2,100)
pdf_values = npdf(x_values,3,2)
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()

# 2. 파이썬 scipy 패키지 사용해서 다음과 같은 확률을 구하시오
# X ~ N(2,3^2)

# 1) P(x<3)
norm.cdf(3,loc=2, scale=3)
# 2) P(2<x<5)
norm.cdf(5,loc=2,scale=3) - norm.cdf(2,2,3)
# 3) P(x<3 or x>7)
norm.cdf(3,2,3) + 1 - norm.cdf(7,2,3)

# 3. LS 빅데이터 스쿨 학생들의 중간고사 점수는 평균 30이고, 분산이 4인 정규분포를 따른다.
# 상위 5%에 해당하는 학생의 점수는?
 
# X ~ N(30, 4=2^2) 
# P(x >= 0.95)
score= norm.ppf(0.95, 30, 2)
score














