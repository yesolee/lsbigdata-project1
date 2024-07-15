# 수학함수

import math

x=4
math.sqrt(x)

math.exp(5)

math.log(10,10)

math.factorial(5)

math.sin(math.radians(90))
math.cos(math.radians(180))
math.tan(math.radians(45))

def normal_pdf(x, mu, sigma):
  part_1 = (sigma * math.sqrt(2*math.pi))**-1
  part_2 = math.exp(-(x-mu)**2 / (2*sigma**2))
  return part_1 * part_2

normal_pdf(3,3,1)


def my_f(x,y,z):
  return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)

my_f(2,9,math.pi/2)

def my_g(x):
  return math.cos(x) + math.sin(x) * math.exp(x)
  
my_g(math.pi)

a=[1,2,3,4]
    
def fname(`indent('.') ? 'self' : ''`):
    """docstring for fname"""
    # TODO: write code...
    
def plus(a):
a=a+a
    return a

import pandas as pd
import numpy as np
def fname(input):
    contents
    return
    
# snippet 단축키 shift+space로 변경했음

import pandas as pd
# !pip install numpy
import numpy as np

a=np.array([1,2,3,4,5]) # 숫자형 벡터 생성
a
type(a)
b=np.array(["apple","banana","orange"]) #문자형 벡터 생성
b
c=np.array([True,False,True,True])
c
a[3] # np.int64(4)
a[2:] # array([3, 4, 5])
a[1:4] # array([2, 3, 4])
b[0] # 
c[2]

b=np.empty(3)
b[0] = 1
b[1] = 2
b[2] = 3
b[2]

vec1 = np.array([1,2,3,4,5])
vec1 = np.arange(1,101,0.5)
vec1

l_space1 = np.linspace(0,1,5)
l_space1

l_space2 = np.linspace(0,1,5, endpoint=False)
l_space2

?np.linspace

np.repeat(3,5)
vec1=np.arange(5)
vec1 # array([0, 1, 2, 3, 4])
np.repeat(vec1,5) # array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4,
       4, 4, 4])
np.tile(vec1,3) # array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
vec1 + vec1 # 벡터가 지원되기 때문에 같은 위치의 값끼리 더해짐

vec2 = np.arange(0,-101,-1)
vec3 = -np.arange(0,101) # 벡터가 지원되기 때문에
vec4 = np.repeat([1,2,3],3)
vec4
vec1
max(vec1)
sum(vec1)

#35672이하 홀수들의 합은?
np.sum(np.arange(1,35673,2))
np.arange(1,35673,2)
x= np.arange(1,35673,2)
x.sum()
np.sum(x)

len(x)
x.shape

[[1,2,3],[4,5,6]]
b = np.array([[1,2,3],[4,5,6]])
b
len(b) # 첫번째 차원의 길이
b.shape # 각 차원의 크기
b.size # 전체 요소의 개수

a=np.array([1,2])
b=np.array([1,2,3,4])
# a+b 길이가 맞지 않으면 덧셈이 안된다.

import numpy as np

a=np.array([1,2])
b=np.array([1,2,3,4])
b
np.title(a,2) +b
np.repeat(a,2)+ b
b == 3

# 35672보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?
sum(np.arange(1,35672) % 7 == 3)


# 10보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?
np.sum((np.arange(1,35672) % 7) == 3)
