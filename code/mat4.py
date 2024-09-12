# y = (x-2)^2 +1 그래프 그려보기

import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-4,8.1,0.1)
x
f_k = (k-2)**2 + 1
plt.plot(k,f_k, color='black')
2*10-4
2*2.1-4
def my_line(k):
    l_slope = 2*k-4
    l_intercept = f_k - l_slope * k
    y_line = l_slope * k + l_intercept
    plt.plot(k,y_line, color='red')
    plt.plot(x,f_k, color='red')
    plt.xlim(-4,8)
    plt.ylim(0,15)

my_line(4)

y_diff = 4*x - 11
plt.plot(x,y_diff, color='red')
plt.scatter(x=4,y=5)
0.9*16-8

x=10
lstep = np.arange(100,0,-1)*0.1

for i in range(1,100001):
    x -= lstep * (2*x)

x

9-1.2