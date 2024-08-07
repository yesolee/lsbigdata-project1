# 패키지 불러오기
import numpy as np
import pandas as pd

tab3 = pd.read_csv('data/tab3.csv')
tab3

tab1 = pd.DataFrame({"id": np.arange(1,13),
                     "score": tab3['score']})
tab1

tab2 = tab1.assign(gender = ['female']*7 +['male']*5 )
tab2

# 1표본 t검정 (그룹1개)
# 귀무가설 vs 대립가설
# H0: mu=10 vs Ha: mu != 10
# 유의수준 5%로 설정
from scipy.stats import ttest_1samp
result = ttest_1samp(tab1['score'],popmean=10, alternative='two-sided')
t_value = result[0] # t 검정 통계량량
p_value = result[1] # 유의 확률 (p-value), 양쪽(two-sided) 더한 값  
result.df
result.pvalue
result.statistic
tab1['score'].mean()
# 귀무가설이 참일때(모평균mu=10), 표본평균이 11.53이 관찰 될 확률이 6.48%로(유의확률) 유의수준 5%보다 크므로 거짓이라고 보기 힘들다.
# 유의확률이 0.06로 유의수준 0.05보다 크므로로 귀무가설 기각 못함함


# 신뢰구간 구하기
ci = result.confidence_interval(confidence_level=0.95) #신뢰구간
ci[0]
ci[1]

# 2표본 t 검정(그룹2) - 분산 같고, 다를 때
### 분산 같은 경우: 독립2표본 t검정 equal_var=True 
### 분산 다를 경우: 웰치스 t검정 equal_var=False
## 귀무가설 VS 대립가설
## H0: mu_m = mu_f vs Ha: mu_m > mu_f
## 유의수준 1%로 설정, 두 그릅 분산 같다고 가정
tab2
from scipy.stats import ttest_ind
female = tab2[tab2['gender'] == "female"]
male = tab2[tab2['gender'] == "male"]

# alternative="less"의 의미는 대립가설이
# 첫번째 입력 그룹의 평균이 두번째 입력그룹 평균보다 작다
result = ttest_ind(female['score'], male['score'], alternative = "less",equal_var=True)
# 동일: ttest_ind(male['score'],female['score'], alternative = "greater",equal_var=True)

result.statistic
result.pvalue
ci=result.confidence_interval(0.95)
ci[0] # 한쪽이 infinit가 나옴
ci[1]

# 대응표본 t검정(짝지을 수 있는 표본)
## 귀무가설 vs 대립가설
## Ho: mu_before = mu_after vs Ha: mu_after>mu_before
## Ho: mu_d = 0 vs Ha: mu_d > 0
## mu_d = mu_after - mu_before
## 유의수준 1%로 설정
# mu_d에 대응하는 표본으로 변환
tab3
tab3_data= tab3.pivot_table(index='id',columns='group', values='score').reset_index()
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
tab3_data = tab3_data[['score_diff']]
tab3_data # score_diff열을 보면 1표본 t테스트가 됨


from scipy.stats import ttest_1samp
result = ttest_1samp(tab3_data['score_diff'],popmean=0, alternative='two-sided')
## Ho: mu_before = mu_after vs Ha: mu_after>mu_before 이기 때문에 popmean=0으로 설설정
t_value = result[0] # t 검정 통계량
p_value = result[1] # 유의 확률 (p-value), 양쪽(two-sided) 더한 값  

df = pd.DataFrame({
    "id":[1,2,3],
    "A":[10,20,30],
    "B":[40,50,60]
})

df

df_long=df.melt(id_vars = 'id',
        value_vars=['A','B'],
        var_name='group',
        value_name='score')
df_long

df_long.pivot_table(
    columns="group",
    values="score",
    # aggfunc='mean' 기본값
)

df_long.pivot_table(
    columns="group",
    values="score",
    aggfunc='max' 
)


import seaborn as sns

tips = sns.load_dataset('tips')
tips

# index를 지정 안해주면 같은 분류로 보고 값이 1개로 나옴
tips.pivot_table(columns='day',values='tip')



# pivot은 중복값은 제외시킨다.
# 요일별로 팁값을 펼치고 싶은 경우
index_list = list(tips.columns.delete(4))
tips.reset_index(drop=False)\
    .pivot_table(index=['index'],columns='day',values='tip')\
    .reset_index()
# tips.pivot(columns='day',values='tip')

tips






