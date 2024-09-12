import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# 펭귄 분류 문제
# y: 펭귄의 종류
# x1: bill_length_mm (부리 길이)
# x2: bill_depth_mm (부리 깊이)

df=penguins.dropna()
df=df[["species", "bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={
    'species': 'y',
    'bill_length_mm': 'x1',
    'bill_depth_mm': 'x2'})
df


# x1, x2 산점도를 그리되, 점 색깔은 펭귄 종별 다르게 그리기!
import seaborn as sns

sns.scatterplot(data=df, x="x1", y="x2", hue='y')
plt.axvline(x=45)

# Q. 나누기 전 현재의 엔트로피?
# Q. 45로 나눴을때, 엔트로피 평균은 얼마인가요?
# 입력값이 벡터 -> 엔트로피!
p_i=df['y'].value_counts() / len(df['y'])
entropy_curr=-sum(p_i * np.log2(p_i))

# x1=45 기준으로 나눈 후, 평균 엔트로피 구하기!
# x1=45 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
n1=df.query("x1 < 45").shape[0]  # 1번 그룹
n2=df.query("x1 >= 45").shape[0] # 2번 그룹

# 1번 그룹은 어떤 종류로 예측하나요?
# 2번 그룹은 어떤 종류로 예측하나요?
y_hat1=df.query("x1 < 45")['y'].mode()
y_hat2=df.query("x1 >= 45")['y'].mode()

# 각 그룹 엔트로피는 얼마 인가요?
p_1=df.query("x1 < 45")['y'].value_counts() / len(df.query("x1 < 45")['y'])
entropy1=-sum(p_1 * np.log2(p_1))

p_2=df.query("x1 >= 45")['y'].value_counts() / len(df.query("x1 >= 45")['y'])
entropy2=-sum(p_2 * np.log2(p_2))

entropy_x145=(n1 * en|tropy1 + n2 * entropy2)/(n1 + n2)
entropy_x145


# x1 기준으로 최적 기준값은 얼마인가?




# 기준값 x를 넣으면 MSE값이 나오는 함수는?
def my_entropy(x):
    n1=df.query(f"x < {x}").shape[0]  # 1번 그룹
    n2=df.query(f"x >= {x}").shape[0] # 2번 그룹
    y_hat1=df.query(f"x < {x}").mean()[0]
    y_hat2=df.query(f"x >= {x}").mean()[0]
    mse1=np.mean((df.query(f"x < {x}")["y"] - y_hat1)**2)
    mse2=np.mean((df.query(f"x >= {x}")["y"] - y_hat2)**2)
    return float((mse1* n1 + mse2 * n2)/(n1+n2))

my_mse(15)
my_mse(13.71)
my_mse(14.01)

df["x"].min()
df["x"].max()

# 13~22 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용해서 가장 작은 MSE가 나오는 x 찾아보세요!
x_values=np.arange(13.2, 16.4, 0.01)
nk=x_values.shape[0]
result=np.repeat(0.0, nk)
for i in range(nk):
    result[i]=my_mse(x_values[i])

result
x_values[np.argmin(result)]
# 14.01, 16.42, 19.4

# x, y 산점도를 그리고, 빨간 평행선 4개 그려주세요!
import matplotlib.pyplot as plt

df.plot(kind="scatter", x="x", y="y")
thresholds=[14.01, 16.42, 19.4]
df["group"]=np.digitize(df["x"], thresholds)
y_mean=df.groupby("group").mean()["y"]
k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.42, 100)
k3=np.linspace(16.42, 19.4, 100)
k4=np.linspace(19.4, 22, 100)
plt.plot(k1, np.repeat(y_mean[0],100), color="red")
plt.plot(k2, np.repeat(y_mean[1],100), color="red")
plt.plot(k3, np.repeat(y_mean[2],100), color="red")
plt.plot(k4, np.repeat(y_mean[3],100), color="red")