import numpy as np

a = np.arange(1,4)
a # [1,2,3]
b = np.array([3,6,9])

a.dot(b) # 1*3 + 2*6 + 3*9

a=np.array([1,2,3,4]).reshape((2,2), order="F")
a

b=np.array([5,6]).reshape(2,1)
b

a.dot(b)

a@b

a=np.array([1,2,3,4]).reshape((2,2), order="F")
a

b=np.array([5,6,7,8]).reshape((2,2), order="F")
b

a@b

a = np.array([1,2,1,0,2,3]).reshape(2,3)
a
b = np.array([1,0,-1,1,2,3]).reshape(3,2)
b

a@b

a = np.array([3,5,7,
              2,4,9,
              3,1,0]).reshape(3,3)
a @ np.eye(3)
np.eye(3) @ a

a
a.transpose()
b = a[:,0:2]
b
b.transpose()
b

a = np.array([1,2,3])
b = np.array([1,2,3]).reshape(3,1)
b
a@b

# 회귀분석 행렬
x = np.array([13,15,
              12,14,
              10,11,
              5,6]).reshape(4,2)

x
vec1= np.repeat(1,4).reshape(4,1)
vec1
matX = np.hstack((vec1, x))
matX

beta_vec = np.array([2,3,1]).reshape(3,1)
beta_vec

matX @ beta_vec

y = np.array([20,19,20,12]).reshape(4,1)
y
# 역행렬

a = np.array([1,5,3,4]).reshape(2,2)
a_inv = (-1/11) * np.array([4,-5,-3,1]).reshape(2,2)
a@a_inv

# 3*3 역행렬
a = np.array([-4,-6,2,
               5,-1,3,
               -2,4,-3]).reshape(3,3)
a
a_inv= np.linalg.inv(a)
a_inv

np.round(a @ a_inv, 3)

### 주의! 역행렬은 항상 존재하는게 아니다!!!! 
## 행렬의 세로 벡터들이 선형독립일때만 역행렬을 구할 수 있다.

# 선형 독립이 아닌 경우(=선형종속) :역행렬 존재 X
# 1 2 3 => 1 1*2 1*2+1
# 2 4 5 => 2 2*2 2*2+1
# 3 6 7 => 3 3*2 3*2+1
# 1,2,3인 1열만 알아도, 2열, 3열 구할 수 있음: 행렬이 singular(특이행렬)

b = np.array([1,2,3,
              2,4,5,
              3,6,7]).reshape(3,3)
b
b_inv= np.linalg.inv(b) # 에러남
b_inv
np.linalg.det(b) # 0이나오면 역행렬 못구하는 애들임
np.linalg.det(a) # 행렬식 ex. 2*2행렬에서는 ad-bc determinant

# 베타 구하기
matX
y
# 방법1 벡터형태로 베타구하기 (수식)
XtX_inv = np.linalg.inv(matX.transpose() @ matX)
Xty = matX.transpose() @ y
beta_hat = XtX_inv @ Xty
beta_hat

# 방법2 : 모델fit으로 베타 구하기
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(matX[:,1:],y)
model.coef_
model.intercept_

# 방법3 minimize로 베타구하기
from scipy.optimize import minimize

def line_perform(beta):
    beta = np.array(beta).reshape(3,1)
    a = (y - matX @ beta)
    return (a.transpose() @ a )

line_perform([8.55, 5.96, -4.38])

# 초기 추정값
initial_guess = [0,0,0]

# 최솟값 찾기
result = minimize(line_perform, initial_guess)

# 최솟값
result.fun

# 최솟값을 갖는 x 값
result.x


# 방법4 minimize로 라쏘 베타구하기
def line_perform_lasso(beta):
    beta = np.array(beta).reshape(3,1)
    a = (y - matX @ beta)
    return (a.transpose() @ a + 3*np.abs(beta).sum())

line_perform([8.55, 5.96, -4.38])
line_perform([3.76, 1.36, 0])
line_perform_lasso([8.55, 5.96, -4.38])
line_perform_lasso([3.76, 1.36, 0]) - 3 * (np.abs(np.array([3.76, 1.36, 0]).sum()))

# 초기 추정값
initial_guess = [0,0,0]

# 최솟값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 최솟값
result.fun

# 최솟값을 갖는 x 값
result.x


# 방법5 minimize로 릿지 베타구하기
def line_perform_ridge(beta):
    beta = np.array(beta).reshape(3,1)
    a = (y - matX @ beta)
    return (a.transpose() @ a + 3*(beta**2).sum())
# norm 2

line_perform_ridge([8.55, 5.96, -4.38])
line_perform_ridge([3.76, 1.36, 0])

# 초기 추정값

initial_guess = [0,0,0]

# 최솟값 찾기
result = minimize(line_perform_ridge, initial_guess)

# 최솟값
result.fun

# 최솟값을 갖는 x 값
result.x

###############

import numpy as np

# 벡터 * 벡터 (내적)
a = np.arange(1, 4)
b = np.array([3, 6, 9])

a.dot(b)

# 행렬 * 벡터 (곱셈)
a = np.array([1, 2, 3, 4]).reshape((2, 2),
                                   order='F')
a

b = np.array([5, 6]).reshape(2, 1)
b

a.dot(b)
a @ b

# 행렬 * 행렬
a = np.array([1, 2, 3, 4]).reshape((2, 2),
                                   order='F')
b = np.array([5, 6, 7, 8]).reshape((2, 2),
                                   order='F')
a
b
a @ b

# Q1.
a = np.array([1, 2, 1, 0, 2, 3]).reshape(2, 3)
b = np.array([1, 0, -1, 1, 2, 3]).reshape(3, 2)

a @ b

# Q2
np.eye(3)
a=np.array([3, 5, 7,
            2, 4, 9,
            3, 1, 0]).reshape(3, 3)

a @ np.eye(3)
np.eye(3) @ a

# transpose
a
a.transpose()
b=a[:,0:2]
b
b.transpose()

# 회귀분석 데이터행렬
x=np.array([13, 15,
           12, 14,
           10, 11,
           5, 6]).reshape(4, 2)
x
vec1=np.repeat(1, 4).reshape(4, 1)
matX=np.hstack((vec1, x))
y=np.array([20, 19, 20, 12]).reshape(4, 1)
matX

beta_vec=np.array([2, 0, 1]).reshape(3, 1)
beta_vec
matX @ beta_vec

(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec)

# 역행렬
a=np.array([1, 5, 3, 4]).reshape(2, 2)
a_inv=(-1/11)*np.array([4, -5, -3, 1]).reshape(2, 2)

a @ a_inv

## 3 by 3 역행렬
a=np.array([-4, -6, 2,
            5, -1, 3,
            -2, 4, -3]).reshape(3,3)
a_inv=np.linalg.inv(a)
np.linalg.det(a)
a_inv

np.round(a @ a_inv, 3)

## 역행렬 존재하는 않는 경우(선형종속)
b=np.array([1, 2, 3,
            2, 4, 5,
            3, 6, 7]).reshape(3,3)
b_inv=np.linalg.inv(b) # 에러남
np.linalg.det(b) # 행렬식이 항상 0

# 벡터 형태로 베타 구하기
matX
y
XtX_inv=np.linalg.inv((matX.transpose() @ matX))
Xty=matX.transpose() @ y
beta_hat=XtX_inv @ Xty
beta_hat

# 모델 fit으로 베타 구하기
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(matX[:, 1:], y)

model.intercept_
model.coef_

# minimize로 베타 구하기
from scipy.optimize import minimize

def line_perform(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a)

line_perform([ 8.55,  5.96, -4.38])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta).sum()

line_perform([8.55,  5.96, -4.38])
line_perform([3.76,  1.36, 0])
line_perform_lasso([8.55,  5.96, -4.38])
line_perform_lasso([3.76,  1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# minimize로 릿지 베타 구하기
from scipy.optimize import minimize

def line_perform_ridge(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 3*(beta**2).sum()

line_perform_ridge([8.55,  5.96, -4.38])
line_perform_ridge([3.76,  1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_ridge, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
