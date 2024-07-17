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
   
   
# 1. 변수 이름 변경 했는지?
# 2. 행들을 필터링 했는지?
# 3. 새로운 변수를 생성했는지?
# 4. 그룹 변수 기준으로 요약을 했는지?
# 5. 정렬 했는지?

# .reset_index() 시리즈를 데이터프레임으로 바꿔줌

import pandas as pd

# 데이터 합치기

test1 = pd.DataFrame({
    'id':[1,2,3,4,5],
    'midterm':[60,80,70,90,85]
})

test2 = pd.DataFrame({'id':[1,2,3,40,5],
'final' :[70,83,65,95,80]})

test1
test2

total_l = pd.merge(test1, test2, how='left', on='id')
total_l # 기준이 test1이기 때문에 40번은 빠짐

total_r = pd.merge(test1, test2, how='right', on='id')
total_r # 기준이 test2이기 때문에 40번은 빠짐

total_i = pd.merge(test1, test2, how='inner', on="id")
total_i # 공통된거

total_o = pd.merge(test1, test2, how='outer', on="id")
total_o # 싹 다


name=pd.DataFrame({'nclass':[1,2,3,4,5],
                   'teacher':['kim','lee','park','choi','jung']})
name

exam = pd.read_csv('data/exam.csv')
exam.head()
exam_t = pd.merge(exam,name,how='left',on='nclass')
exam_t.head()

# 데이터를 세로로 쌓는 방법 (칼럼이 똑같아야함)

score1 = pd.DataFrame({
    'id':[1,2,3,4,5],
    'score':[60,80,70,90,85]
})

score2 = pd.DataFrame({'id':[6,7,8,9,10],
'score' :[70,83,65,95,80]})

score1
score2

score_all = pd.concat([score1, score2])
score_all = pd.concat([score1, score2], ignore_index = True)
score_all


test1
test2
pd.concat([test1,test2],axis=1)
