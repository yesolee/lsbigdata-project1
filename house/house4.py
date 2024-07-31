import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
    

# 파일 경로는 항상 project working directory 기준!!
house_train = pd.read_csv('house/train.csv')
house_train.info()

GarageType = house_train.groupby('GarageType').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
#house_train.groupby('GarageYrBlt').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
GarageFinish = house_train.groupby('GarageFinish').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
GarageQual = house_train.groupby('GarageQual').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
# GarageCond = house_train.groupby('GarageCond').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
PavedDrive = house_train.groupby('PavedDrive').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)

# house_train.groupby('Condition1').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
# house_train.groupby('Condition2').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
plt.figure(figsize=(10, 6))

sns.lineplot(data=GarageFinish)
plt.show()
plt.clf()

sns.lineplot(data=GarageQual)
plt.show()
plt.clf()

sns.lineplot(data=GarageType)
plt.show()
plt.clf()

sns.lineplot(data=PavedDrive)
plt.show()
plt.clf()

GarageType = house_train.groupby('GarageType',as_index=False).agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
GarageType
#house_train.groupby('GarageYrBlt').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
GarageFinish = house_train.groupby('GarageFinish',as_index=False).agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
GarageFinish
GarageQual = house_train.groupby('GarageQual',as_index=False).agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
GarageQual
# GarageCond = house_train.groupby('GarageCond').agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)
PavedDrive = house_train.groupby('PavedDrive',as_index=False).agg(mean=('SalePrice','mean')).sort_values('mean',ascending=False)

correlation = GarageType["GarageType"].corr(GarageType["mean"])

quality_mapping = {
    'Ex': 5,
    'Gd': 4,
    'Ta': 3,
    'Fa': 2,
    'Po': 1
}

GarageType["GarageType"] = GarageType["GarageType"].replace(quality_mapping)


house_test = pd.read_csv('house/test.csv')
house_test = house_test[['Id','GarageFinish','GarageQual','GarageType']]
house_test


house_test = pd.merge(house_test, GarageType, how="left", on = "GarageType")
house_test
house_test = house_test.rename(columns = {'mean':'SalePrice'})
house_test

house_test['SalePrice'].isna().sum()

# 비어있는 테스트 세트 집들 확인
house_test.loc[house_test['SalePrice'].isna()]

# 집값 채우기
house_mean = house_train['SalePrice'].mean
house_test['SalePrice'] = house_test['SalePrice'].fillna(house_mean)
house_test.isna().sum()

# sub 데이터 불러오기
house_sub = pd.read_csv('house/sample_submission.csv')
house_sub 

# SalePrice 바꿔치기
house_sub['SalePrice']= house_test['SalePrice']
house_sub

house_sub.to_csv('house/sample_submission10.csv', index=False)






