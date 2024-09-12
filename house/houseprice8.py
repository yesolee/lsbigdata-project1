import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("train.csv")
house_test=pd.read_csv("test.csv")
sub_df=pd.read_csv("sample_submission.csv")

df = pd.concat([house_train, house_test], ignore_index=True)
df

neighborhood_dummies = pd.get_dummies(
    df["Neighborhood"],
    drop_first=True
    )
neighborhood_dummies

x= pd.concat([df[["GrLivArea", "GarageArea"]], 
             neighborhood_dummies], axis=1)
y = df["SalePrice"]

train_x = x.iloc[:1460,]
train_x

test_x = x.iloc[1460:,]
test_x

train_y = y[:1460]
train_y

# 모의고사 셋 만들기 (validation)
np.random.seed(42)
val_index = np.random.choice(np.arange(1460),size = 438, replace=False)
val_index

valid_x = train_x.iloc[val_index] # 30%
valid_x

train_x = train_x.drop(val_index) # 70%

valid_y = train_y[val_index]
valid_y

