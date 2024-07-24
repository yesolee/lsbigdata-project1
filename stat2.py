import numpy as np
import matplotlib.pyplot as plt

# 예제넘파이 배열 생성
# data = np.random.rand(10)
# 히스토램(빈도표) 그리기
# bins=4의 의미 : 0에서부터 1사이를 4개 구간으로 쪼개서 보여줘
# alpha=0.7의 의미: 막대 색을 살짝 연하게 
plt.hist(data, bins= 4, alpha=1, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
plt.clf()

data = np.random.rand(50000).reshape(-1,5)
# np.random.rand(10000,5) 5개씩 만번 뽑아줘 해도 동일함
data
mean_data=data.mean(axis=1)
plt.hist(mean_data, bins= 30, alpha=0.7, color='blue')
plt.show()
plt.clf()

import os
print(os.getcwd()) # 현재경로

import pandas as pd

score1 = pd.DataFrame({"id"     : [1, 2, 3, 4, 5], 
                      "score": [60, 80, 70, 90, 85]})

score2 = pd.DataFrame({"id"     : [6, 7, 8, 9, 10],
                      "score"  : [70, 83, 65, 95, 80]})
score1
score2
score_all=pd.concat([score1, score2], axis=1)
score_all

test1
test2
b
pd.concat([test1, test2], axis=1)

np.arange(33).sum()/33
sum(np.unique((np.arange(33) - 16)**2))*2/33
np.unique((np.arange(33) - 16)**2)
x=np.arange(33)
sum(x**2)/33 - 16**2


x= np.array([0,1,2,3])
x
x**2
pro_x= np.array([1,2,2,1])/6
pro_x
Exx = sum((x**2)*pro_x)
Ex= sum(x*pro_x)
# 분산 
# 방법1) Exx - Ex**2
Exx - Ex**2

# 방법2) E[(x - Ex)**2]
sum( ((x-Ex)**2) * pro_x)

np.arange(99)
import numpy as np

x = np.arange(99)
x
# 1-50-1 벡터
x_1_50_1 =np.concatenate((np.arange(1,51),np.arange(49,0,-1))) 
pro_x = x_1_50_1/2500
pro_x

Ex = sum(x*pro_x)
Exx= sum(x**2 * pro_x)

Exx - Ex**2
sum((x- Ex)**2 * pro_x)

y=np.arange(4)+3
y
pro_y=np.array([1,2,2,1])/6
pro_y
Ey = sum(y*pro_y)
Eyy = sum(y**2 * pro_y)dq

Eyy - Ey**2

sum((y-Ey)**2 * pro_y)

0.916
8.25/0.916

3.625/5

# !pip install scipy
from scipy.stats import bernoulli
#확률 질량 함수 pmf(k,p) : 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
bernoulli.pmf(1, 0.3)

bernoulli.pmf(0, 0.3)
