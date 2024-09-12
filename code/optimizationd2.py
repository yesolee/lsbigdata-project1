import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis')

# 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Graph of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# ==========================
# 등고선 그래프
import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (9, 2)에 파란색 점을 표시
plt.scatter(9, 2, color='red', s=50)

x=9; y=2
lstep=0.1
for i in range(100):
    x, y = np.array([x, y]) - lstep * np.array([2*x-6, 2*y-8])
    plt.scatter(float(x), float(y), color='red', s=50)
print(x,y)

# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# 그래프 표시
plt.show()

##########################
x = np.arange(-15,15)
y = np.arange(-15,15)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = 4*x**2 + 20*x*y + 30*y**2 -23*x -67*y + 44.25

plt.figure()
cp = plt.contour(x, y, z, levels=100)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (9, 2)에 파란색 점을 표시



plt.scatter(x, y, color='blue', s=5)
# 그래프 표시
plt.show()

# 모델fit으로 베타 구하기
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    'x': np.array([1,2,3,4]),
    'y': np.array([1,4,1.5,5])
})
model = LinearRegression()
model.fit(df[['x']],df[['y']])

model.intercept_
model.coef_


df = pd.DataFrame({
    '부리길이y':[13,15,17,15,16], 
    '날개길이x1':[16,20,22,18,17], 
    '부리깊이x2':[7,7,5,6,7] 
}) 

df
# 날개길이(x1) 15, 부리깊이(x2) 5.6인 펭귄 부리길이(y)는?
# 예측값은 얼마?

(13+15+16)/3

# KNN 회귀모델

# K-nearse nighborhood

