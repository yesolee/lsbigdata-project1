# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 라쏘, 릿지 회귀분석 => 범주형 더미코딩해서 하면 변수가 너무많아... 뭐슨 칼럼이 최적일까?찾는 모델

## 필요한 데이터 불러오기
blueberry_train=pd.read_csv("train.csv")
blueberry_test=pd.read_csv("test.csv")
sub_df=pd.read_csv("sample_submission.csv")

## 결측값 확인
blueberry_train.describe()
blueberry_test.info()
blueberry_train.isna().sum()
blueberry_test.isna().sum()

# ## 이상치 제거

# for i in blueberry_train.drop(['id','yield'], axis=1):

#     # IQR 계산
#     Q1 = blueberry_train[i].quantile(0.25)
#     Q3 = blueberry_train[i].quantile(0.75)
#     IQR = Q3 - Q1
#     print(IQR)

#     # 이상치 조건 설정
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     # 이상치 제거
#     blueberry_train = blueberry_train[(blueberry_train[i] >= lower_bound) & (blueberry_train[i] <= upper_bound)]

# #########

train_x = blueberry_train.drop(['id','yield'], axis=1)
train_y = blueberry_train['yield']
test_x = blueberry_test.drop(['id'], axis=1)


# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf,
                                     n_jobs=-1, scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장``
alpha_values = np.arange(0.0, 1, 0.01)
nhbh_values = np.arange(1,101)

mean_scores_lasso = np.zeros(len(alpha_values))
mean_scores_ridge = np.zeros(len(alpha_values))
mean_scores_knn = np.zeros(len(nhbh_values))

k=0
for alpha, nhbh in zip(alpha_values,nhbh_values):
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)
    knn = KNeighborsRegressor(n_neighbors=nhbh)
    mean_scores_lasso[k] = rmse(lasso)
    mean_scores_ridge[k] = rmse(ridge)
    mean_scores_knn[k] = rmse(knn)
    k += 1

# 결과를 DataFrame으로 저장
df_lambda = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error_lasso': mean_scores_lasso,
    'validation_error_ridge': mean_scores_ridge,    
})

df_lambda

df_nhbh = pd.DataFrame({
    'nhbh': nhbh_values, 
    'validation_error_knn': mean_scores_knn,  
})
df_nhbh

# 결과 시각화
plt.plot(df_lambda['lambda'], df_lambda['validation_error_lasso'], label='Validation Error_lasso', color='red')
plt.plot(df_lambda['lambda'], df_lambda['validation_error_ridge'], label='Validation Error_ridge', color='blue')
plt.plot(df_nhbh['nhbh'], df_nhbh['validation_error_knn'], label='Validation Error_knn', color='green')
plt.legend()
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha_lasso = df_lambda['lambda'][np.argmin(df_lambda['validation_error_lasso'])]
optimal_alpha_ridge = df_lambda['lambda'][np.argmin(df_lambda['validation_error_ridge'])]
optimal_nhbh_knn = df_nhbh['nhbh'][np.argmin(df_nhbh['validation_error_knn'])]
print("Optimal lambda_lasso:", optimal_alpha_lasso)
print("Optimal lambda_ridge:", optimal_alpha_ridge)
print("Optimal nhbh_knn:", optimal_nhbh_knn)

# 일반회귀로 y 예측하기
model1 = LinearRegression()
model1.fit(train_x, train_y)
pred_y_ols = model1.predict(test_x)

train_x
test_x

# 라쏘로 y 예측하기
model2 = Lasso(alpha=optimal_alpha_lasso)
model2.fit(train_x, train_y)
pred_y_lasso = model2.predict(test_x)

# 릿지로 y 예측하기
model3 = Lasso(alpha=optimal_alpha_ridge)
model3.fit(train_x, train_y)
pred_y_ridge = model3.predict(test_x)

# KNN로 y 예측하기
model4 = KNeighborsRegressor(n_neighbors=10)
model4.fit(train_x, train_y)
pred_y_knn = model4.predict(test_x)

# 가중치
w = [0.25,0.25,0.25,0.25]
pred_y = w[0]*pred_y_ols+w[1]*pred_y_lasso+w[2]*pred_y_ridge+w[3]*pred_y_knn

# yield 바꿔치기
sub_df["yield"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("sample_submission_11.csv", index=False)

