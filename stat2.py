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
