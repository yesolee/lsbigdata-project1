def g(x=3):
    result = x+1
    return result

# 함수 내용 확인
import inspect
print(inspect.getsource(g))

import pandas
g(4)

import numpy as np

x= np.array([1,-2,3,-4,0])
conditions = [
    x>0, x==0, x<0
]
choice=["양수","0","음수"]
result = np.select(conditions, choice, x)
result

# for loop
for i in range(1,4):
    print(f"Here is {i}")
    

# for loop 리스트 컴프

name = "John"
age= 30
print(greeting)

names = ["John", "Alice"]
ages = [25,30]

zipped = zip(names, ages)
greetings = [f"Hello, my name is {name} and I am {age} years old." for name, age in zip(names, ages)]
for greeting in greetings:
    print(greeting)
    
i=0
while i<= 10:
    i += 3
    print(i)
    
# while, break문
i=0
while True:
    i += 3
    if i>10:
        break
    print(i)
    
    
import pandas as pd
data = pd.DataFrame({
    'A':[1,2,3],
    'B':[4,5,6]
})

data

data.apply(max, axis=0)
data

data.apply(max, axis=1)

def my_func(x, const=3):
    return max(x)**2 + const

my_func([1,2,3],4)

data.apply(my_func, axis=0, const=3)

array1 = np.arange(1,13).reshape((3,4), order='F')
np.apply_along_axis(max, axis=1, arr = array1)

# 함수환경
y=2
def my_func(x):
    global y
    
    def my_f(k):
        return k**2
    
    y= my_f(x) +1
    result = x+y
    return result

my_func(3)
print(y)

# 입력값이 몇개인지 모를땐 *
def add_many(*args):
    result = 0
    for i in args:
        result = result + 1 
    return result
add_many(1,2,3)

add_many([1,2,3])

def first_many(*args):
    return args[0]

first_many(3,1,2,3)
first_many([3,1,2,3])

def add_mul(choice, *my_input):
    if choice == "add":
        result = 0
        for i in my_input:
            result = result + i
    elif choice == "mul":
        result = 1
        for i in my_input:
            result = result * i
    return result

add_mul("mul",5,4,3,1)
      
## 별표 두개(**)는 입력값을 딕셔너리로 만들어줌!
def my_twostars(choice, **kwargs):
    if choice == "first":
        return kwargs["age"]
    elif choice == "second":
        return kwargs["name"]
    else:
        return kwargs
      
print(my_twostars("all",age=30, name="issac", job="student"))    
