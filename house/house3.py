import pandas as pd

# 파일 경로는 항상 project working directory 기준!!
house_train = pd.read_csv('house/train.csv')
house_train = house_train[['Id','YearBuilt','SalePrice']]
house_train.info()

# 연도별 평균
house_mean = house_train.groupby('YearBuilt', as_index=False)\
                        .agg(mean_year=('SalePrice','mean'))
house_mean

house_test = pd.read_csv('house/test.csv')
house_test = house_test[['Id','YearBuilt']]
house_test

house_test = pd.merge(house_test, house_mean, how="left", on="YearBuilt")
house_test = house_test.rename(columns = {'mean_year':'SalePrice'})
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
sub_df

sub_df.to_csv('house/sample_submission.csv', index=False)
house_df['YearBuilt']





