import pandas as pd
import numpy as np

# 데이터 탐색 함수
# head()
# tail()
# shape()
# info()
# describe()

exam = pd.read_csv('data/exam.csv')
exam.head(10)
exam.tail(10)
exam.shape
exam.info()
exam.describe()
# 메서드(함수) vs 어트리뷰트(속성)

type(exam)
var=[1,2,3]
type(var)
exam.head()
#var.head()

exam2 = exam.copy() 
exam2.rename(columns={'nclass':'class'}) # 원본은 바뀌지 않음
exam2 = exam2.rename(columns={'nclass':'class'})
exam2.head()
exam2['total'] = exam2['math']+exam2['english']+exam2['science']
exam2.head()

exam2['test'] = np.where(exam2['total']>=200,'pass','fail')
exam2.head()

import matplotlib.pyplot as plt
count_test=exam2['test'].value_counts()
?count_test.plot.bar
(rot=0)
plt.show()
plt.clf() # 데이터 지워줌. plt.show를 하면 데이터 지워져서 안보임

?DataFrame.plot

exam2['test2'] = np.where(exam2['total']>=200,'A',
                np.where(exam2['total']>=100,'B','C'))
exam2.head()
exam2['test2'].value_counts()


exam2['test2'].isin(['A','C']).value_counts()

