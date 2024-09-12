# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 라쏘, 릿지 회귀분석 => 범주형 더미코딩해서 하면 변수가 너무많아... 뭐슨 칼럼이 최적일까?찾는 모델

## 필요한 데이터 불러오기
house_train=pd.read_csv("train.csv")
house_test=pd.read_csv("test.csv")
sub_df=pd.read_csv("sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## train 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

## test 숫자형 채우기
quantitative_test = house_test.select_dtypes(include = [int, float])
quant_selected_test = quantitative_test.columns[quantitative_test.isna().sum() > 0]

for col in quant_selected_test:
    house_test[col].fillna(house_test[col].mean(), inplace=True)

## train 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)

## test 범주형 채우기
qualitative_test = house_test.select_dtypes(include = [object])
qual_selected_test = qualitative_test.columns[qualitative_test.isna().sum() > 0]

for col in qual_selected_test:
    house_test[col].fillna("unknown", inplace=True)    

train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )

# x,y나누기
df_x = df.drop(columns=['Id','SalePrice'])
df_y = df['SalePrice']

# train / test 데이터셋
train_x=df_x.iloc[:train_n,]
train_y=df_y.iloc[:train_n,]

test_x=df_x.iloc[train_n:,]

# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(50, 500, 1)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df_lambda = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df_lambda

# 결과 시각화
plt.plot(df_lambda['lambda'], df_lambda['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df_lambda['lambda'][np.argmin(df_lambda['validation_error'])]
print("Optimal lambda:", optimal_alpha)


# 회귀분석 데이터행렬

vec1=np.repeat(1, len(train_x)).reshape(-1, 1)
matX=np.hstack((vec1, train_x))

# 모델 fit으로 베타 구하기
model = Lasso(alpha=optimal_alpha)
model.fit(matX[:, 1:], train_y)

model.intercept_
model.coef_

# SalePrice 바꿔치기
pred_y = model.predict(test_x)
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("sample_submission_lasso2.csv", index=False)

