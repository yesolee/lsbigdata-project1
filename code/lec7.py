# 한 리스트에 여러 타입을 담을 수 있다
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]

# 빈 리스트 생성
a=list()
a
b=[]
b

# 초기 값을 가진 리스트 생성
numbers=[1,2,3,4,5]
numbers2=list(range(10))
numbers2

numbers[3] ="빅데이터 스쿨"
numbers

numbers[1] = ["1st","2nd","3rd"]
numbers[1][2]

# 리스트 컴프리헨션
# 1. 대괄호로 쌓여져있다 => 리스트다
# 2. 넣고 싶은 수식 표현을 x를 사용해서 표현
# 3. for .. in .. 을 사용해서 원소 정보 제공
squares = [x**2 for x in range(10)]
squares

list(range(10))

my_squares= [x**3 for x in [3,5,2,15]]
my_squares

# 넘파이 어레이도 가능
import numpy as np
my_squares= [x**3 for x in np.array([3,5,2,15])]
my_squares

# 판다스 시리즈도 가능
import pandas as pd
exam = pd.read_csv('data/exam.csv')
type(exam['math'])
my_squares= [x**3 for x in exam['math']]
my_squares

# 리스트 합치기
3+2
"안녕"+"하세요"
"안녕"*3
# 리스트 연결
list1=[1,2,3]
list2=[4,5,6]
combined_list=list1+list2
list1 * 3
(list1*3)+(list2*5)

numbers = [5,2,3]
# python에서 사용하는 _ 의 의미
# 1) 앞에서 나온 값을 가리킬때 사용
5+4
_+6 # _는 9를 의미
# 2) 값 생략, 자리차지
a,_,b = (1,2,4)
a
b
_
_ = None
del _
repeated_list = [x for x in numbers for _ in [4,2,1,3]]
repeated_list = [(x,y) for x in numbers for y in range(4)]

# for 루프 문법
# for i in 범위 :작동
#   작동방식

for x in [4,1,2,3]:
    print(x)

for i in range(5):
    print(i**2)
    
# 리스트를 하나 만들어서 
# for 루프를 사용해서 2,4,6,8,...20의 수를 채워넣어 보세요!

my_list = []
for i in range(2,21,2):
    my_list.append(i)

my_list2 = [0]*10
for i in range(11):
    my_list2[i] = 2*(i+1)
my_list2

[i for i in range(2,21,2)]

# 인덱스 공유해서 카피하기
mylist_b = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
mylist = [0]*10
for i in range(10):
    mylist[i] = mylist_b[i]
mylist[2] = 100
mylist
mylist_b

# Quiz: mylist_b의 홀수번째 위치의의 숫자들만 
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0]*5
for i in range(5):
    mylist[i] = mylist_b[2*i]
mylist

# 리스트 컴프리헨션으로 바꾸는 방법]
# 바깥은 무조건 대괄호로 묶어줌: 리스트로 반환하기 위해서
# for 루프의 :는 생략한다.
# 실행할 부분을 먼저 써준다.
# 결과값을 발생하는 표현만 남겨두기

[ i*2 for i in range(1,11)]
[ i for i in [0,1] for _ in [4,5,6]]
for i in [0,1]:
    for j in [4,5,6]:
        print(i)
        
# 리스트 컴프리헨션 변환
[i for i in numbers for j in range(4)]
for x in numbers:
    for y in range(4):
        print(x,y)
        
        
repeated_list

a=[]
a.append([2,4,5])
a

i=2
my_list

## 원소 체크
fruits = ["apple","banana","cherry"]
[x == "banana" for x in fruits]
mylist=[]
for x in fruits:
    mylist.append(x == "banana")
mylist

# banana가 위치한 인덱스
fruits = ["apple","apple","banana","cherry"]
my_index=0
for i in range(len(fruits)):
    if fruits[i] == "banana":
        my_index = i
my_index


# np로 바나나의 위치를 뱉어내게 하려면?
import numpy as np
# np를 쓰려면 np 자료형으로 바꿔줘야 한다!
fruits = np.array(fruits)
np.where(fruits == "banana") # 튜플플
np.where(fruits == "banana")[0][0]

# 원소 거꾸로 써주는 reverse()
fruits = ["apple","apple","banana","cherry"]
fruits.reverse()
fruits

fruits.append("pineapple")
fruits
fruits.reverse()
fruits

# 원소 삽입
fruits.insert(2,'test')
fruits
fruits.insert(0,'test2')
fruits

# 원소 제거
fruits.remove('apple')
fruits

fruits.insert(2,'apple')
fruits



fruits = np.array(fruits)
fruits[~np.isin(fruits,['banana','apple'])]
