#4. vector-in-python 중 11p broadcasting

a=np.array([1.0,2.0,3.0])
b=2.0
a.shape # (3,)
b.shape # float이라 shape 존재하지 않음
a*b # array([2., 4., 6.])


matrix = np.array([[ 0.0,  0.0,  0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
matrix.shape
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4,1)
vector
vector.shape
print(vector.shape)
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

# 넘파이 벡터 슬라이싱
import numpy as np
?np.random.seed
np.random.seed(2024)
test= np.random.choice(np.arange(1,4) ,100, True, np.array([2/5,2/5,1/5]))
test # 확률 합이 1이 되야함
sum(test == 1)
sum(test == 2)
sum(test == 3)

?np.random.randint
a = np.random.randint(1,21,10)
a 
print(a[1]) # 두번째 값 추출

a[2:5]
a[-1] # 맨 끝에서 두번째
a[-2]
a[::2] # 처음부터 끝까지 스텝은2
a[:] # 처음부터 끝까지

1에서부터 1000사이 3의 배수의 합은?
sum(np.arange(3,1001)[::3])
sum(np.arange(0,1001,3))
sum(np.arange(3,1001,3))

a
print(a[[0,2,4]])
np.delete(a,3) # a의 3번인덱스 삭제한 값을 리턴 (원본 수정X)
a
np.delete(a, [1,3,5]) # a의 1번,3번,5번 인덱스 값 삭제
a
a>3
a[a>3]

np.random.seed(2024)
a = np.random.randint(1,10000,5)
(a>2000) & (a<5000)
a[a<5000]

a[(a>2000)&(a<5000)]

#!pip install pydataset
import pydataset
df= pydataset.data('mtcars')
np_df = np.array(df['mpg'])

model_names = np.array(df.index)
model_names

# 15이상 25이하인 데이터 개수는?
sum( (np_df >= 15) & (np_df <=25) )
# 평균 mpg보다 높은(이상) 자동차 모델은?
model_names[ np_df >= np.mean(np_df) ]
model_names[ np_df < np.mean(np_df) ]
# 15이상 20이하인 데이터 개수는?
model_names[(np_df < 15) | (np_df>=22)] 


np.random.seed(2024)
a= np.random.randint(1,10000,5)
a
b= np.array(["A","B","C","F","W"])
b
# a[조건을 만족하는 논리형 벡터]
a[(a>2000)&(a<5000)]
b[(a>2000)&(a<5000)] # a가 True인 위치를 b에 적용

a[a>3000] = 3000 # a>3000 이상인 값에 3000을 대입
a

np.random.seed(2024)
a = np.random.randint(1,100,10)
a
a<50
np.where(a<50) # a<50을 만족하는 인덱스 
# (np.where-> 조건을만족하는인덱스반환)
a[np.where(a<50)]

np.random.seed(2024)
a=np.random.randint(1,26346,1000)
a

#처음으로 22000보다 큰 숫자? 위치와 그 숫자
x = np.where(a>22000)
x[0][0]
type(x)
my_index=x[0][0]
a[my_index]
type(x[0])
a[a>22000][0]
np.where(a>10000)[0][0]
a[np.where(a>10000)[0]][0]

# 처음으로 10000보다 큰 숫자 나왔을때, 50번째 숫자 위치와 숫자
x= np.where(a>10000)
x
my_index = x[0][49]
my_index # 81
a[my_index] # 21052

# 500보다 작은 숫자들 중 
# 가장 마지막으로 나오는 숫자 위치와 그 숫자
x= np.where(a<500)
x
my_index = x[0][-1]
my_index # 960
a[my_index] # 391

np.nan+3 

a= np.array([20, np.nan, 13, 24, 309])
a
a[~np.isnan(a)]
np.isnan(a)
sum(np.isnan(a))
a+3
np.nan+3
np.mean(a)
np.nanmean(a) # nan무시하고 평균내기 \
np.array([20,13,24,309]).mean()
?np.nan_to_num
np.nan_to_num(a, nan = 0)
a

None
a=None
b=np.nan
a+1
b+1

np.isnan(a)
sum(np.isnan(b)) # np.True_
arr=[1,2,3,4,5]
arr[[0]]
arr[[0,2]]

a = np.array([1, 2, 3, 4, 16, 17, 18]) 
a
a[0]
a[0,2]
a[[0]]
a[[0,2]]
str_vec = np.array(["사과","배", "수박", "참외"])
str_vec
str_vec[0] #np.str_('사과')
str_vec[[0]] #array(['사과'], dtype='<U2')
str_vec[[0,2]] # array(['사과', '수박'], dtype='<U2')

mix_vec = np.array(["사과", 12, "배", "수박", "참외"], dtype=str)
mix_vec # array(['사과', '12', '배', '수박', '참외'], dtype='<U2')
# 문자열로 바뀜
c = np.array(["1","2","3","4","5"])
c
#숫자열로 바뀜
c = np.array(["1","2","3","4","5"], dtype=int)
c
 
comboned_vec = np.concatenate((str_vec, mix_vec))
comboned_vec 

col_stacked = np.column_stack((np.arange(1,5),np.arange(12,16)))
col_stacked

row_stacked = np.row_stack((np.arange(1,5),
np.arange(12,16)))
row_stacked

vec1 = np.arange(1,5)
vec2 = np.arange(12,18)

vec1 = np.resize(vec1, 15)
vec1

a= np.array([1,2,3,4,5])
a
a+5

a=np.array([12,21,35,48,5])
a[::2]

a = np.array([1, 22, 93, 64, 54])
a
a.max()

a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
a
np.unique(a)
 
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])

x=np.empty(6)
x

# 짝수번쨰
# x[[1,3,5]] = b
x[1::2] = b
#x[[0,2,4]]=a
x
x[::2] = a
x
x[:3]

## 교재 77페이지


