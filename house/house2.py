import pandas as pd

# 파일 경로는 항상 project working directory 기준!!
house_train = pd.read_csv('house/train.csv')
house_train.info()

# 연도별 평균
house_Tot = house_train.groupby(['LotArea','TotRmsAbvGrd','YearBuilt','GarageFinish'], as_index=False).agg(mean_Tot=('SalePrice','mean'))
house_Tot
house_test = pd.read_csv('house/test.csv')
house_test

house_test = pd.merge(house_test, house_Tot, how="left", on=['LotArea','TotRmsAbvGrd','YearBuilt','GarageFinish'])
house_test = house_test.rename(columns = {'mean_Tot':'SalePrice'})
house_test

house_test['SalePrice'].isna().sum()

# 비어있는 테스트 세트 집들 확인
house_test.loc[house_test['SalePrice'].isna()]

# 집값 채우기
house_mean = house_train['SalePrice'].mean()
house_test['SalePrice'] = house_test['SalePrice'].fillna(house_mean)
house_test.isna().sum()

# sub 데이터 불러오기
house_sub = pd.read_csv('house/sample_submission.csv')
house_sub 

# SalePrice 바꿔치기
house_sub['SalePrice']= house_test['SalePrice']
house_sub

house_sub.to_csv('house/sample_submission6.csv', index=False)





