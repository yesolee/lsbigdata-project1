import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

df = penguins.dropna()
df = df.rename(columns={'bill_length_mm':'y',
    'bill_depth_mm':'x'}
)
df = df[['y','x']]
df

# 기준값x를 넣으면 mse값이 나오는 함수는?
def my_mse(x):
    group1 = df.query(f'x < {x}')
    group2 = df.query(f'x >= {x}')
    group1.mean()[0]
    group2.mean()[0]

    # 각 그룹 MSE = 평균제곱오차
    mse1=((group1['y']-group1.mean()[0])**2).mean()
    mse2=((group2['y']-group2.mean()[0])**2).mean()

    # x=20의 MSE가중평균은?
    result = (mse1*len(group1) + mse2*len(group2) ) / len(df) 
    return round(result,3)

x_value = np.arange(13.2,21.4,0.01)
result=[]

for i in x_value:
    result.append(my_mse(i))

result
result_first = x_value[np.argmin(result)]
result_first #16.40999

df_left = df.query(f'x<{result_first}')
df_right = df.query(f'x>{result_first}')
df_right['x'].min()
df= df_left
result_left =[]
x_value_left = np.arange(13.2,result_first,0.01)
for i in x_value_left:
    result_left.append(my_mse(i))

result_second = x_value_left[np.argmin(result_left)]
result_second # 14.0099


df = df_right
result_right =[]
x_value_right = np.arange(df_right['x'].min()+0.01,21.4,0.01)
x_value_right
for i in x_value_right:
    result_right.append(my_mse(i))

result_third = x_value_right[np.argmin(result_right)]
result_third # 19.40


penguins = load_penguins()
penguins.head()

df = penguins.dropna()
df = df.rename(columns={'bill_length_mm':'y',
    'bill_depth_mm':'x'}
)
df = df[['y','x']]
df

import matplotlib.pyplot as plt

df_1 = df.query(f'x<{result_second}')
df_2 = df.query(f'x>{result_second} and x<{result_first}')
df_3 = df.query(f'x>{result_first} and x<{result_third}')
df_4 = df.query(f'x>{result_third}')

plt.scatter(df['x'],df['y'])
plt.axvline(x=result_first, color='black', linestyle=':', linewidth=1)
plt.axvline(x=result_second, color='black', linestyle=':', linewidth=1)
plt.axvline(x=result_third, color='black', linestyle=':', linewidth=1)

k1 = np.linspace(df['x'].min(),result_second,100)
k2 = np.linspace(result_second,result_first,100)
k3 = np.linspace(result_first, result_third,100)
k4 = np.linspace(result_third, df['x'].max(),100)
plt.plot(k1, [df_1['y'].mean()]*len(k1), color='red')
plt.plot(k2, [df_2['y'].mean()]*len(k2), color='red')
plt.plot(k3, [df_3['y'].mean()]*len(k3), color='red')
plt.plot(k4, [df_4['y'].mean()]*len(k4), color='red')