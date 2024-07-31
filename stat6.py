import numpy as np
from scipy.stats import norm
# X ~N(3,7^2)
x = norm.ppf(0.25, loc =3, scale=7 )

# X ~N(0,1^2) 표준정규분포
z = norm.ppf(0.25, loc =0, scale=1 )
z*7+3

norm.cdf(5,loc=3,scale=7)
norm.cdf(2/7,loc=0,scale=1)
norm.ppf(0.975,loc=0,scale=1)
norm.ppf(0.025,loc=0,scale=1)

#표준정규분포에서 표존 1000개 꺼내서 히스토그램 => pdf 겹처서그리기
z = norm.rvs(loc=0, scale=1, size=1000)
z
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(z,stat="density", color="grey")
zmin, zmax= (z.min(), x.max()) # x축 범위까지 Z의 분포를 보여주고 싶어서 
z_values=np.linspace(zmin,zmax,500)
z_pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, z_pdf_values, color='red', linewidth=2)

# Z를 X ~ N(3,sqrt(2)^2)로 바꾸고 싶어
x = z*np.sqrt(2) +3
x
sns.histplot(x,stat="density", color="blue")
xmin, xmax= (x.min(), x.max())
x_values=np.linspace(xmin,xmax,100)
x_pdf_values = norm.pdf(x_values, loc=3, scale=np.sqrt(2))
plt.plot(x_values, x_pdf_values, color='black', linewidth=2)
plt.show()
plt.clf()

# X~N(5,3^2) 일떄 
# z=(x-5)/3 가 표준 정규분포를 따르나요? (표준화 확인인)
x=norm.rvs(loc=5, scale=3, size=1000)
#표준화
z=(x-5)/3
#z의 히스토그램그리기
sns.histplot(z,stat="density", color="salmon")
# 표준정규분포(N(0,1))그리고 겹치는기 확인하기
zmin, zmax= (z.min(), z.max()) # x축 범위까지 Z의 분포를 보여주고 싶어서 
z_values=np.linspace(zmin,zmax,100)
z_pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, z_pdf_values, color='red', linewidth=2)

# 표본보준편차로 나눠도 표준정규분포가 될까?
# X~N(5,3^2) 일떄 
# 1. 표본을 10개 뽑아 표본 분산값 계산
x10=norm.rvs(loc=5, scale=3, size=20)
s=np.std(x10, ddof=1)
s
# 3은 아니지만 비슷하니까 갖다 쓰자 => 그래도 정규분포로 갈까?
# 잘맞을때도 있고, 완전 안맞을떄도 있고! 20개만 뽑았기떄문에??
# 결론: 안맞다 

# 2. X표본 1000개 뽑음
x1000=norm.rvs(loc=5, scale=3, size=1000)
# 3. 1번에서 계산한 s^2로 sigma^2를 대체한 표준화를 진행 z=(x-mu)/s
z2=(x1000-5)/s
# 4. z의 히스토그램 구하기
sns.histplot(z2,stat="density", color="grey")
plt.show()
plt.clf()


# t분포
# X ~ t(df) 모수가 1개짜리 분포, 연속형, 종모양, 대칭, 중심이 0으로 정해짐
# 모수 df: 자유도라고 부름 - 퍼짐을 나타내는 모수
# 자유도 :  분산(그래프의 넓이)를 조절하는 모수
# df가 작으면 분산 커짐
from scipy.stats import t

t.pdf(df)
t.ppf
t.cdf
t.rvs

# 자유도가 4인 t분포의 pdf를 그려보세요!
# 끝단이 발생 많이하면 꼬리가 길다고 이야기 한다.(빨간색)
t_values=np.linspace(-4,4,100)
pdf_values = t.pdf(t_values, df=4)
plt.plot(t_values, pdf_values, color='red', linewidth=2)
z_values=np.linspace(-4,4,100)
z_pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, z_pdf_values, color='black', linewidth=1)

# df가 4->1로 줄으면 꼬리가 더 두터워짐 fat tail
t_values=np.linspace(-4,4,100)
pdf_values = t.pdf(t_values, df=1)
plt.plot(t_values, pdf_values, color='blue', linewidth=2)

# df(n)가 4->2로 줄으면 꼬리가 더 두터워짐 fat tail
t_values=np.linspace(-4,4,100)
pdf_values = t.pdf(t_values, df=2)
plt.plot(t_values, pdf_values, color='green', linewidth=2)

# 자유도 30 => 표준정규분포랑 거의 가까워짐
# n이 무한대로 가면 표준 정규분포가 된다.
# 분포를 추정할떄 샘플의 수가 적을수록 차이가 많이 난다.
t_values=np.linspace(-4,4,100)
pdf_values = t.pdf(t_values, df=30)
plt.plot(t_values, pdf_values, color='pink', linewidth=2)

plt.show()
plt.clf()

# X ~ ?(mu, sigma)
# X bar ~ N(mu, sigma^2/n)
# X bar ~= t(x_bar, s^2/n)
# sigma를 s로 대체했기 떄문에 정규분포를 따르는게 아니라 자유도가 n-1인 t로 따르게 된다.
x = norm.rvs(loc=15, scale=3, size=16, random_state=42)
x
x_bar = x.mean()
n=len(x)

# df = degree of freedom
# 모분산을 모를때 : 모평균에 대한 95% 신뢰구간 구하기
x_bar + t.ppf(0.975, df=n-1) * np.std(x,ddof=1)/np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x,ddof=1)/np.sqrt(n)

# 모분산을 알때 : 모평균에 대한 95% 신뢰구간 구하기
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3/np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, scale=1) * 3/np.sqrt(n)















