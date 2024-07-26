import numpy as np

from scipy.stats import uniform

# X ~ 균일분포 U(a,b)
# loc = a, scale = b-a
# uniform 에서 loc은 구간 시작점, scale=길이이

uniform.rvs(loc=2, scale=4, size=1)
k=np.linspace(0,8,100)
k
y=uniform.pdf(k, loc=2, scale=4)
plt.plot(k,y,color="black")
plt.show()
plt.clf()
(3.25-2)*0.25

uniform.cdf(3.25,loc=2,scale=4)
uniform.cdf(8.39,loc=2,scale=4) - uniform.cdf(5,loc=2,scale=4)
uniform.ppf(0.93,loc=2,scale=4)

# 표본 20개를 뽑아서 표본평균을 구하시오
x = uniform.rvs(loc=2, scale=4, size = 20, random_state = 42)
x # scipy 해도 x출력해보면 np로 나오기 때문에 random_Stat대신 seed써도 됨
x.mean()
# 여러개 만들고 싶으면면
y = uniform.rvs(loc=2, scale=4, size = 20*1000, random_state = 42)
y.shape
y = y.reshape(-1,20)
y.shape
blue_x = y.mean(axis=1)
blue_x.shape
import seaborn as sns
sns.histplot(blue_x, stat="density")
plt.show()
plt.clf()

# X bar~ N(mu,sigma&2/n) / X ~ U(2,6)의 mean=4니까
# X bar~ N(3, 1.3333/20)
blue_x.mean()
uniform.var(loc=2, scale=4) # 검은색 벽돌을 발생시키는 분산
uniform.expect(loc=2, scale=4) # 검은색 벽돌을 발생시키는 기댓값(평균)

# 
from scipy.stats import norm
xmin, xmax= (blue_x.min(), blue_x.max())
x_values=np.linspace(xmin,xmax,100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()



# 신뢰구간
# X bar ~N(mu, sigma^2/n)
# X bar ~N(4, 1.33333/20)
from scipy.stats import norm

x_values=np.linspace(3,5,100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)

a = norm.ppf(0.025, loc=4, scale=np.sqrt(1.3333/20))
b = norm.ppf(0.975, loc=4, scale=np.sqrt(1.3333/20))
4-a
b-4
0.506/ np.sqrt(1.3333/20)
np.sqrt(1.3333/20)
# 표본평균(파란 벽돌) 점찍기
# norm.ppf(0.975, loc=0, scale=1) == 1.96
blue_x = uniform.rvs(loc=2, scale=4, size=20).mean()
a= blue_x +1.96 * np.sqrt(1.3333/20)
b= blue_x -1.96 * np.sqrt(1.3333/20)
plt.scatter(blue_x, 0.002, color='blue', zorder=10, s=10)

# 기댓값 표현
plt.axvline(x=4, color='green', linestyle='-', linewidth=2)
plt.axvline(x=a, color='blue', linestyle='-', linewidth=1)
plt.axvline(x=b, color='blue', linestyle='-', linewidth=1)
plt.show()
plt.clf() 

# 95% 해당하는 a,b 확인
norm.ppf(0.025, loc=4, scale= np.sqrt(1.3333333/20))
norm.ppf(0.975, loc=4, scale= np.sqrt(1.3333333/20))
