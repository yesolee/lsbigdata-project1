import pandas as pd
import numpy as np

df = pd.DataFrame({
    'sex': ['M','F', np.nan, 'M', 'F'],
    'score': [5,4,3,4,np.nan]
    })
df

df['score']+1

pd.isna(df)
pd.isna(df).sum()

# 결측치 제거하기
df.dropna(subset='score')
df.dropna(subset=['score','sex'])
df.dropna()
df.loc[df["score"]==4,["score"]]=100
df

# 데이터 프레임 location을 사용한 인덱싱
# exam.loc[행 인덱스, 열 인덱스]
exam = pd.read_csv('data/exam.csv')
exam.loc[[2,7,14],]
exam.loc[[0],['id','nclass']]
exam.iloc[0:2,0:4]
exam.iloc[0:2]

# 수학 점수가 50점 이하인 학생들 점수 50점으로 상향 조정
exam.loc[exam['math']<=50, 'math'] = 50
exam
# exam['math']= np.where(exam['math']<=50,50,exam['math'])

# 영어 점수가 90점 이상인 학생들 점수, 90점으로 하향조점 iloc사용
exam.loc[exam['english']>=90,'english']

# iloc을 사용해서 조회하려면 무조건 숫자 벡터가 들어가야함.
exam.iloc[exam['english']>=90,3]  # 실행안됨

exam.iloc[np.array(exam['english']>=90),3] # 실행됨
exam.iloc[np.where(exam['english']>=90)[0],3] # np.where도 튜플이라 [0]로 np.array를 꺼내오면 됌
exam.iloc[exam[exam['english']>=90].index,3] # index 벡터도 작동

type(exam[exam['english']>=90].index)
# <class 'pandas.core.indexes.base.Index'>
exam.head()
# math 점수 50점 이하 -로 변경
exam.iloc[np.array(exam['math']<=50),2]='-'
exam
mean_math = exam.query('math != "-"')['math'].mean()
m_math    = exam.loc[(exam['math']!= "-"),'math'].mean()
me_math   = exam.query('math not in ["-"]')['math'].mean()
mea_math  = np.nanmean(np.array([np.nan if x == '-' else float(x) for x in exam['math']]))

import numpy as np

type(exam['math'])

vector1 = np.array([np.nan if x == '-' else float(x) for x in exam['math']])
vector2 = np.array([float(x) if x != "-" else np.nan for x in exam['math']])
vector2
# 

exam.loc[exam['math']=="-",['math']]=np.nan
exam
exam.loc[pd.isna(exam['math']),['math']]

exam.iloc[np.array(exam['math']=="-"),2]=mean_math
exam['math'] = np.where(exam['math'] == "-", math_mean, exam['math'])
exam['math'] = np.where(exam['math] == "-", exam['math'])
exam['math'] = exam['math'].replace("-", math_mean)


