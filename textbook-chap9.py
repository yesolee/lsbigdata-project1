import pandas as pd
import numpy as np
import seaborn as sns

# !pip install pyreadstat
raw_welfare= pd.read_spss('./koweps/Koweps_hpwc14_2019_beta2.sav')
raw_welfare

welfare = raw_welfare.copy()

welfare
welfare.shape
welfare.info()
welfare.describe()

welfare = welfare.rename(
    columns = {
        "h14_g3": "sex",
        "h14_g4": "birth",
        "h14_g10": "marriage_type",
        "h14_g11": "religion",
        "p1402_8aq1": "income",
        "h14_eco9": "code_job",
        "h14_reg7": "code_region",
    }
)

welfare = welfare[["sex","birth","marriage_type","religion","income","code_job","code_region"]]
welfare.shape

welfare['sex'].dtypes
welfare['sex'].value_counts()
welfare['sex'].isna().sum()

welfare['sex'] = np.where(welfare['sex']==1,'male','female')
welfare['sex']

welfare['income'].describe()
welfare['income'].isna().sum()
welfare['income'].value_counts().sort_values()
sum(welfare['income'] >9998)

sex_income = welfare.dropna(subset='income').groupby('sex', as_index=False).agg(mean_income=("income","mean"))
sex_income

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data=sex_income, x="sex", y="mean_income", hue="sex")
plt.show()
plt.clf()

# 숙제 
# 위 그래프에서 각 성별 95% 신뢰구간 계산 후 그리기
# 위 아래 검성색 막대기로 표시
## 1. 평균구하기
mean_women = sex_income.query('sex == "female"')['mean_income']
mean_man = sex_income.query('sex == "male"')['mean_income']

# 235p
welfare['birth'].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
plt.clf()

welfare['birth'].describe()
welfare["birth"].isna().sum()

welfare = welfare.assign(age=2019 - welfare['birth'] + 1)
welfare['age']

sns.histplot(data=welfare, x='age')
plt.show()
plt.clf()

age_income = welfare.dropna(subset='income').groupby('age').agg(mean_income=('income','mean'))
age_income.head()

sns.lineplot(data=age_income,x='age', y='mean_income')
plt.show()
plt.clf()

my_df = welfare.assign(income_na=welfare['income'].isna())\
               .groupby('age', as_index=False)\
               .agg(n=('income_na','sum'))
my_df
sns.barplot(data=my_df, x='age',y='n')
plt.show()
plt.clf()


welfare['age'].head()
welfare = welfare.assign(ageg= np.where(welfare['age']<30, 'young',
                               np.where(welfare['age']<=59, 'middle',
                                                            'old')))
welfare['ageg'].value_counts()
                                                        
sns.countplot(data = welfare, x= 'ageg')
plt.show()
plt.clf()

ageg_income= welfare.dropna(subset='income').groupby('ageg', as_index=False).agg(mean_income=('income','mean'))
sns.barplot(data = ageg_income, x='ageg', y='mean_income',hue='ageg')
sns.barplot(data=ageg_income, x='ageg', y='mean_income', order=['young', 'middle', 'old'],hue='ageg')
plt.show()
plt.clf()

#np.array([0])+[i*10+9 for i in np.arange(max(welfare['age'])//10+1)]
# cut
bin_cut = np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])
pd.cut(vec_x, bins = bin_cut)
welfare = welfare.assign(
    age_group = pd.cut(welfare['age'],
                       bins = bin_cut,
                       labels=(np.arange(12)*10).astype(str) + "대"))

age_income=welfare.dropna(subset='income')\
                  .groupby('age_group', as_index=False)\
                  .agg(mean_income=('income','mean'))

age_income
sns.barplot(data=age_income, x='age_group', y='mean_income')
plt.show()
plt.clf()

?np.array
# 판다스 데이터를 다룰 때, 변수의 타입이 카테고리로 설정되어 있는 경우, groupby+agg콤보 안먹힘
# object로 타입 변경 후 수행!
welfare['age_group'] = welfare['age_group'].astype('object')
welfare['age_group']
# x에는 'income'이 들어간다
sex_income = welfare.dropna(subset='income')\
                    .groupby(['age_group','sex'],as_index=False)\
                    .agg(top4per=('income',lambda x: np.quantile(x, q=0.96)))
sex_income

sns.barplot(data= sex_income, x='age_group', y='mean_income',hue='sex')
plt.show()
plt.clf()

# 연령대별,성별별 상위 4% 수입 찾아보세요!
sex_income_top4 = \
    welfare.dropna(subset='income')\
           .groupby(['age_group','sex'],as_index=False).agg(top4per_income=('income', lambda x: np.quantile(x,q=0.96)))
sex_income_top4

sns.barplot(data=sex_income_top4,
            x="age_group", y="top4per_income",hue="sex")
plt.show()
plt.clf()


## 참고
welfare.dropna(subset="income").groupby('sex')[['income']].mean()
welfare.dropna(subset="income").groupby('sex')[['income']].agg(['std','mean'])
welfare.dropna(subset="income").groupby('sex')['income'].agg(['std','mean'])

welfare['code_job']
welfare['code_job'].value_counts()

# !pip install openpyxl
list_job=pd.read_excel('./koweps/Koweps_Codebook_2019.xlsx', sheet_name='직종코드')
list_job.head()


welfare=welfare.merge(list_job,how='left',on='code_job')
job_income=welfare.dropna(subset=['job','income'])\
                  .query("sex=='female'")\
                  .groupby('job')\
                  .agg(mean_income=('income','mean'))
job_income
top10 = job_income.sort_values('mean_income',ascending=False).head(10)
top10

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':5})

sns.barplot(data=top10, y='job', x='mean_income',hue='job')
plt.tight_layout()
plt.show()
plt.clf()

# 263p 9-8
welfare['marriage_type']
df=welfare.query("marriage_type!=5")\
          .groupby('religion', as_index=False)\
          ['marriage_type']\
          .value_counts(normalize=True)# 핵심!

df.query('marriage_type==1')\
  .assign(proportion = df['proportion']*100)\
  .round(1)






