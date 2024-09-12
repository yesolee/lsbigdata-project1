import pandas as import pandas as pd

# 워킹 디렉토리 설정(내 깃허브 보고 누가 쓰더라도 쉽게 파일 불러올 수 있게)
import os
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

admission_data = pd.read_csv('./data/admission.csv')
admission_data.shape

# GPA : 학점
# GRE : 대학원 입학시험(영어,수학)

# 합격을 한 사건: Amdit일때, Amdit의 확률 오즈(Odds)는?
# P(Admit) = 합격인원/전체학생
p_hat = admission_data['admit'].mean()
p_hat/(1-p_hat)

# p(A):0.5 -> 확률의 오즈비1
# p(A)작아질수록-> 오즈비 0에가까워짐
# p(A)커질수록->무한대에 가까워짐
# 확률의 오즈비가 갖는 값의 범위: 0~무한대

# 오즈비가 3이면 P(A)는?

# admission 데이터 산점도그리기
# x:gre, y=admission

import seaborn as sns
sns.scatterplot(admission_data, x='gre', y='admit')

!pip install statsmodels
import numpy as np
log_odds = (-3.4075)+0*(-0.0576)+450*0.0023+3*0.7753+2*(-0.5614)
odds = np.exp(log_odds)
odds
p_hat = odds/(odds+1)
p_hat