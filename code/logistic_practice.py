# 종속변수: 백혈병 세포 관측 불가 여부 (REMISS), 1이면 관측 안됨을 의미
# 독립변수:
# 골수의 세포성 (CELL)
# 골수편의 백혈구 비율 (SMEAR)
# 골수의 백혈병 세포 침투 비율 (INFIL)
# 골수 백혈병 세포의 라벨링 인덱스 (LI)
# 말초혈액의 백혈병 세포 수 (BLAST)
# 치료 시작 전 최고 체온 (TEMP)

# 문제 1.
# 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('../data/leukemia_remission.txt', delim_whitespace=True)
df
model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP", data=df).fit()
model.summary()
# 문제 2.
# 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.
# LLR p-value:	0.04670 0.05보다 작으므로 모델이 통계적으로 유의하다.(귀무가설:통계적으로 유의하지 않다, 0.05이하 귀무가설 기각)

# 문제 3.
# 유의수준이 0.2일떄 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?
# P>|z|가 0.2보다 작은 LI, TEMP가 유의하다

# 문제 4. 다음 환자에 대한 오즈는 얼마인가요?
# CELL (골수의 세포성): 65%
# SMEAR (골수편의 백혈구 비율): 45%
# INFIL (골수의 백혈병 세포 침투 비율): 55%
# LI (골수 백혈병 세포의 라벨링 인덱스): 1.2
# BLAST (말초혈액의 백혈병 세포 수): 1.1 세포/μL
# TEMP (치료 시작 전 최고 체온): 0.9
import numpy as np
odds = np.exp(64.2581+ 0.65*30.8301+0.45*24.6863+0.55*-24.9745+1.2*4.3605+1.1*-0.0115+0.9*-100.1734)
odds # 0.03817459641135519

# 문제 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
p_hat = odds/(odds+1)
p_hat

# 문제 6. TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.
# TEMP 변수의 계수: -100.1734
round(np.exp(-100.1734),4) # 0
# 지수 값이 거의 0에 가까우므로, TEMP가 증가할수록 백혈병 치료 성공 가능성(REMISS = 1)이 급격히 감소한다. 즉, 백혈병 치료에 대한 음의 영향을 미친다.

# 문제 7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.
from scipy.stats import norm
z005 = norm.ppf(0.995) 
z005 # 2.5758293035489004
30.8301 - z005 * 52.135  # -103.4607607405219
30.8301 + z005 * 52.135  # 165.12096074052192

# 문제 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
df
x = df.drop(['REMISS'], axis=1)
df['pred_y'] = round(model.predict(x),4)
df['y_hat'] = np.where(df['pred_y']>=0.5,1,0)
df.info()

from sklearn.metrics import confusion_matrix

conf_mat=confusion_matrix(y_true=df['REMISS'], 
                          y_pred=df['y_hat'],
                          labels=[1, 0])

conf_mat

from sklearn.metrics import ConfusionMatrixDisplay

p=ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                         display_labels=("1", "0"))
p.plot(cmap="Blues")


# 문제 9. 해당 모델의 Accuracy는 얼마인가요?
Accuracy = (5+15)/len(df) 
Accuracy # 0.7407407407407407

# 문제 10. 해당 모델의 F1 Score를 구하세요.
Precision = 5/(5+3) # 0.625
Recall = 5/(5+4) # 0.5555555555555556
F1_score = 2*Precision*Recall/(Precision+Recall)
F1_score # 0.5882352941176471

