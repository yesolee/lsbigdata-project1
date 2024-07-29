import pandas as pd

# 파일 경로는 항상 project working directory 기준!!
house_df = pd.read_csv('house/train.csv')
house_df.shape
price_mean = house_df['SalePrice'].mean()
price_mean

sub_df = pd.read_csv('house/sample_submission.csv')
sub_df

sub_df['SalePrice']= price_mean
sub_df

sub_df.to_csv('house/sample_submission.csv', index=False)
house_df['YearBuilt']

house_df.groupby('YearBuilt').
