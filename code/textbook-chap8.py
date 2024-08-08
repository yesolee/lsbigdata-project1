# 교재 8장, p.212

import pandas as pd
economics = pd.read_csv('./data/economics.csv')
economics

economics.info()

import seaborn as sns
sns.lineplot(data= economics, x='date', y='unemploy')
import matplotlib.pyplot as plt
plt.show()
plt.clf()
economics['date2'] = pd.to_datetime(economics['date'])
economics.info()
economics.head()
economics[['date','date2']]
economics['date2'].dt.year
economics['date2'].dt.month
economics['date2'].dt.day
economics['date2'].dt.month_name()
economics['date2'].dt.quarter
economics['quarter']=economics['date2'].dt.quarter
economics['year'] = economics['date2'].dt.year
economics[['date2','quarter']]

#각 날짜가 무슨 요일인가?
economics['date2'].dt.day_name()
economics['date2'].dt.day_of_week
economics['date2'].dt.day_of_year

# 날짜 더해주기: economics['date2'] + 3?
economics['date2'] + pd.DateOffset(days=30)
economics['date2'] + pd.DateOffset(months=1)
economics['date2'] + pd.DateOffset(years=1)
economics['date2'].dt.is_leap_year #윤년(2월29일) 체크
economics.head()

# 216 p 그래프
sns.lineplot(data = economics, x='year', y='unemploy')
sns.scatterplot(data = economics, x='year', y='unemploy', s=2)
plt.show()


#각 연도에 표본평균이랑 표본편차 구하기
my_df = economics.groupby('year', as_index=False).agg(
    mon_mean = ('unemploy','mean'),
    mon_std = ('unemploy','std' ),
    mon_n = ('unemploy','count')
)
# z를 90% 1.96으로 통일
# mean + 1.96 * std / sqrt(n)
# 217p 그래프프
my_df['left_ci'] = my_df['mon_mean'] - 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])
my_df['right_ci'] = my_df['mon_mean'] + 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])
my_df

x=my_df['year']
y=my_df['mon_mean']
plt.plot(x,y)
plt.scatter(x,my_df['left_ci'],color='red', s=1)
plt.scatter(x,my_df['right_ci'],color='black', s=1)
plt.show()
plt.clf()


economics.info


