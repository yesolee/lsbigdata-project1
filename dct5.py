import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# 펭귄 분류 문제
# y: 펭귄의 종류
# x1: 부리길이
# x2: 부리깊이

df=penguins.dropna()
df=df[["species","bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={
    'species':'y',
    'bill_length_mm': 'x1',
    'bill_depth_mm': 'x2'})
len(df['x1'].unique())
df['x1'].describe()
# 기준값 x를 넣으면 엔트로피값이 나오는 함수는?
len(df[df['x1']<42.309])
len(df[df['x1']>=42.309])

def my_entropy(x):
    df_left = df[df['x1']<x]
    df_right = df[df['x1']>=x]

    p_i = df_left['y'].value_counts() / len(df_left['y'])
    entropy_next_left = -sum(p_i * np.log2(p_i))

    p_i = df_right['y'].value_counts() / len(df_right['y'])
    entropy_next_right = -sum(p_i * np.log2(p_i))

    entropy_next = len(df_left)/len(df) * entropy_next_left + len(df_right)/len(df) * entropy_next_right
    return entropy_next

# x1 기준으로 최적 기준값은 얼마인가?
x_range = np.arange(df['x1'].min(),df['x1'].max(),0.01)
x_range
x_unique = sorted(df['x1'].unique())
list_entropy = []
for i in x_range:
    list_entropy.append(my_entropy(i))
    
list_entropy_unique = []
for i in x_unique:
    list_entropy_unique.append(my_entropy(i))
    
best_x_unique = x_unique[np.argmin(list_entropy_unique)]

best_x = x_range[np.argmin(list_entropy)]
best_x # 42.30999999999797
best_x_unique # 42.4 
my_entropy(best_x) # 0.8042691471139144
my_entropy(best_x_unique) # 0.8042691471139144

## 값은 같으나 arange로 하는게 데이터 값에 좌지우지 되기보다는 일정하게 볼수있어서 더 낫다???
## 성능 차이가 크지 않기 때문에 range로해라