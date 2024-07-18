# lec 6 행렬

import numpy as np

matrix = np.vstack(
    (np.arange(1,5),
    np.arange(12,16))
)
matrix

type(matrix)
print("행렬:\n", matrix)

np.zeros(5)
np.zeros([5,4])
np.arange(1,7).reshape((2,3))
np.arange(1,7).reshape((2,-1)) # -1로 하면 3을 알아서 채워줌
# -1 통해서 크기를 자동으로 결정할 수 있음.

# Q. 0에서 99까지 수 중 랜덤하게 50개 숫자를 뽑아서 
# 5 by 10 행렬을 만드세요. (정수)
np.random.seed(2024)
np.random.randint(0,100,50).reshape((5,-1)) # 1열이 아닌경우 튜플로 넣는게 정석
np.arange(1,21).reshape((4,5))
mat_a = np.arange(1,21).reshape((4,5), order='F') # 세로로 값을 채워줌

# 인덱싱
mat_a[1,1] 
mat_a[2,3] 
mat_a[0:2,3]
mat_a[1:3,1:4]

# 행자리, 열자리 비어있는 경우 전체 행, 또는 열 선택
mat_a[3,]
mat_a[3,:]
mat_a[3,::2]
mat_a[1::2,]
mat_a[,3]
mat_a[:,3]

# 짝수 행만 선택하려면?
mat_b= np.arange(1,101).reshape((20,-1))
mat_b[행,열]
mat_b[1::2,:]
mat_b[[1,4,6,14],]

import numppy as np
x= np.arange(1,11).reshape(5,2)*2 
filtered_elements = x[[True,True,False,False,True],0]

mat_b[:,1] # 벡터(1차원)
mat_b[:,1].reshape((-1,1))
mat_b[:,(1,)] # 행렬(2차원) = 매트릭스
mat_b[:,[1]]
mat_b[:,1:2]

# 필터링
mat_b= np.arange(1,101).reshape((20,-1))
mat_b
mat_b[:,1] # 인덱스1 열
# True, False벡터로 행을 골라 낼 수 있다
mat_b[mat_b[:,1]%7 == 0,:]  # 인덱스1열의 값이 7의 배수인 모든 행

# 사진은 행렬이다
import numpy as np
import matplotlib.pyplot as plt

# 난수 생성하여 3*3 행렬 생성

img1 = np.random.rand(3,3)
img1
print("이미지 행렬 img1:\n",img1)

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
plt.clf()

a=np.random.randint(0,256,20).reshape(4,-1)
a/255
plt.imshow(img1, cmap='gray', interpolation='nearest')

import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

import imageio
# !pip install imageio
jelly = imageio.imread("jelly.png")
jelly.shape

jelly[:,:,0].shape
jelly[:,:,1].shape
jelly[:,:,2].shape
jelly[:,:,3].shape

jelly[:,:,0].transpose().shape

plt.imshow(jelly)
plt.show()
plt.clf()
plt.imshow(jelly[:,:,0]) # R
plt.show()
plt.clf()
plt.imshow(jelly[:,:,1]) # P
plt.show()
plt.clf()
plt.imshow(jelly[:,:,2]) # G
plt.show()
plt.clf()
plt.imshow(jelly[:,:,3]) # 투명도
plt.show()
plt.clf()
plt.axis('off') # 축 정보 없애기
plt.show()

# 3차원 배열
# 두 개의 2*3 행렬 생성
mat1 = np.arange(1,7).reshape(2,3)
mat1
mat2 = np.arange(7,13).reshape(2,3)
mat2
my_array = np.array([mat1,mat2])
my_array.shape
my_array

my_array2 = np.array([my_array, my_array])
my_array2.shape

my_array
filtered_array = my_array[0,1,1:] 
filtered_arrays

mat_x = np.arange(1,101).reshape((5,5,4))
mat_x.shape
mat_y = np.arange(1,101).reshape((-1,5,2))
mat_y

# 넘파이 배열 메서드
a = np.array([[1,2,3],[4,5,6]])
a
a.sum()
a.sum(axis=0)
a.sum(axis=1)

a.mean()
a.mean(axis=0)
a.mean(axis=1)

mat_b = np.random.randint(1,100,50).reshape((5,-1))
mat_b

# 가장 큰 수는?
mat_b.max()
mat_b.max(axis=1)
mat_b.max(axis=0)

a= np.array([1,3,2,5]).reshape((2,2))
a
a.cumsum() #누적 합

# 행 별 누적
mat_b
mat_b.cumsum(axis=1) # 누적합
mat_b.cumprod(axis=1) # 누적 곱

a
a= np.array([1,3,2,5])
a
a.cumprod()
 
a= np.array([1,3,2,5]).reshape((2,2))
a
a.cumprod(axis=1) #누적 합

mat_b.reshape((2,5,-1)).flatten()

d= np.array([1,2,3,4,5])
d.clip(2,4) # 경계값 넘어가지 못하게

type(d.tolist())


