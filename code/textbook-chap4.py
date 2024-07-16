import numpy as np
import pandas as pd

df = pd.DataFrame({
    'name':['김지훈','이유진','박동현','김민지'],
    'english':[90,80,60,70],
    'math':[50,60,100,20]
})
df
type(df) # <class 'pandas.core.frame.DataFrame'>

df["name"]
type(df["name"]) # <class 'pandas.core.series.Series'>

df["english"]
sum(df["english"])/4
df["english"].mean()

fruits = pd.DataFrame({
    '제품':['사과','딸기','수박'],
    '가격':[1800,1500,3000],
    '판매량':[24,38,13]
})
fruits

fruits["가격"].mean()
fruits["판매량"].mean()


df[["name","english"]]
df(["name","english")]
df[("name","english")]

df["name"]

pd.show_versions()

#!pip install openpyxl
df_exam = pd.read_excel('data/excel_exam.xlsx', header=None)
df_exam

?pd.read_excel
sum(df_exam['math'])/20
sum(df_exam['english'])/20
sum(df_exam['science'])/20

df_exam
df_exam.shape # (행,열)
len(df_exam) # 행 갯수
df_exam.size # 전체 요소 갯수

df_exam = pd.read_excel('data/excel_exam.xlsx', sheet_name = "Sheet2")
df_exam

df_exam = pd.read_excel('data/excel_exam.xlsx', sheet_name = "Sheet3")
df_exam

df_exam = pd.read_excel('data/excel_exam.xlsx', sheet_name = "Sheet3", header=None)
df_exam

df_exam = pd.read_excel('data/excel_exam.xlsx')
df_exam

df_exam['math']+df_exam["english"]+df_exam["science"] # 시리즈

df_exam['total'] = df_exam['math']+df_exam["english"]+df_exam["science"]
df_exam.head()

df_exam['mean'] = df_exam['total']/3
df_exam.head()

df_exam['math']>50 # 시리즈
df_exam[df_exam['math']>50]

df_exam[(df_exam['math']>50)&(df_exam['english']>50)]

mean_m = df_exam['math'].mean()
mean_m
mean_e = df_exam['english'].mean()
mean_e
df_exam[(df_exam['nclass']==3)&
        (df_exam['math']>mean_m )&
        (df_exam['english']<mean_e)]


df_exam[df_exam['nclass']==3][["math","english","science"]]
df_nc3 = df_exam[df_exam['nclass']==3]
df_nc3[["math","english","science"]]
df_nc3
df_nc3[::3]

df_exam = pd.read_excel('data/excel_exam.xlsx')
df_exam

df_exam.sort_values(['nclass','math'],ascending=[True, False])

a
a>3
np.where(a>3) 
type(np.where(a>3)) # <class 'tuple'>
np.where(a>3,"up","down") 
type(np.where(a>3,"up","down")) # <class 'numpy.ndarray'>
a

df_exam['math'] > 50
np.where(df_exam['math'] > 50)
df_exam["updown"] = np.where(df_exam['math'] > 50,"up","down")
df_exam
