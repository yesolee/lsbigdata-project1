import pandas as pd

mpg=pd.read_csv('data/mpg.csv')
mpg.shape

import seaborn as sns
import matplotlib.pyplot as plt

# 사이즈 조정
plt.figure(figsize=(5,4))
sns.scatterplot(data=mpg,
                x='displ',y='hwy',
                hue='drv')\
   .set(xlim=[3,6],ylim=[10,30])
plt.show()
plt.clf()

?sns.scatterplot

# 막대그래프
df_mpg=mpg.groupby('drv',as_index=False)\
          .agg(mean_hwy=('hwy','mean'))\
          .sort_values('mean_hwy',ascending=False)
# 유니크한 값만 보고 싶을 떄
mpg['drv'].unique()
#
df_mpg
sns.barplot(data=df_mpg,
            x="drv", y="mean_hwy",
            hue="drv")
plt.show()
plt.clf()

df_mpg = mpg.groupby('drv', as_index= False)\
            .agg(n=('drv','count'))
df_mpg['drv'].unique()
# 0번쨰 열 옆에 y 값이 들어갈 값이 있는 자료가 필요함
sns.barplot(data=df_mpg, x='drv', y='n',hue='drv')
plt.show()
plt.clf()
# raw_data가 들어감감
sns.countplot(data=mpg, x='drv', hue='drv', order=['4','f','r'])
plt.show()
plt.clf()
