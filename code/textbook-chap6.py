import pandas as pd
import numpy as np

# 데이터 전처리 함수
# query() :조건에 맞는 행을 걸러냄
# df[]
# sort_values()
# groupby()
# assign()
# agg() : 요약
# merge()
# concat()

exam = pd.read_csv('data/exam.csv')
exam.query('nclass == 1')# exam[exam['nclass'] == 1]와 같음
exam.query('nclass != 1')

exam.query('math>50')
exam.query('math<50')
exam.query('english >=50')
exam.query('english <= 80')
exam.query('nclass == 1 & math >= 50')
exam.query('nclass == 2 & english >= 80')
exam.query('math >=90 | english >= 90')
exam.query('english <90 or science<50')
exam.query('nclass == 1 | nclass ==3 | nclass == 5')
exam.query('nclass in [1,3,5]')
# exam[exam['nclass'].isin([1,3,5])]
exam.query('nclass not in [1,3,5]')
# exam[~exam['nclass'].isin([1,2])]

nclass1 = exam.query('nclass == 1')
nclass1['math'].mean()
nclass2 = exam.query('nclass == 2')
nclass2['math'].mean()

exam['nclass']
exam[['nclass']]
exam[['id','nclass']]
exam.drop(columns=['math','english'])
exam

exam.query('nclass == 1')\
     [['math','english']]\
     .head()

# 정렬하기
exam.sort_values("math",ascending=False).head()
exam.sort_values(['math','english'], ascending=[True,False])

#변수 추가
exam.head()
exam =exam.assign(
    total = exam['math']+exam['english']+exam['science'],
    mean = (exam['math']+exam['english']+exam['science'])/3
    )\
    .sort_values('total', ascending=False)
exam

# lambda 함수 사용하기
exam2 = pd.read_csv('data/exam.csv')
exam2
exam2 =exam2.assign(
    total = lambda x : x['math'] + x['english'] + x['science'],
    mean = lambda x : x['total'] / 3
    )\
    .sort_values('total', ascending=False)
exam2.head()


# 요약을 하는 .agg()
exam2.agg(mean_math= ('math','mean'))

# 그룹을 나눠 요약하는 .groupby() + .agg() 콤보
exam2.groupby('nclass')\
     .agg(mean_math = ('math','mean'))
     
#반별, 과목별 평균
exam2.groupby('nclass')\
     .agg(
         mean_math = ('math','mean'),
         mean_english = ('english','mean'),
         mean_science = ('science','mean')
     )
exam.groupby('nclass').mean()

df_mpg=pd.read_csv('data/mpg.csv')
mpg=df_mpg.copy()
mpg.head()

mpg.query('category=="suv"')\
   .assign(total = (mpg['hwy']+mpg['cty'])/2)\
   .groupby('manufacturer')\
   .agg(mean_tot=('total','mean'))\
   .sort_values('mean_tot', ascending=False)\
   .head()
