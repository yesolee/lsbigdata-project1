# ctrl+Enter:실행
# shift+화살표:블록

# .  현재폴더
# .. 상위폴더

# More > show folder in new window: 해당위치 탐색기
# More > Open new termeinal here 해당 폴더에서 터미널 열기

# 파워쉘 명령어
# cd 폴더명 : 하위 폴더로 이동
# cd 폴더명/폴더명 : 하위 하위 폴더로 이동
# cd ..: 상위 폴더로 이동
# cd ../.. : 2개 폴더 위로 이동
# cd 폴더명 첫글자 tab : 이름 자동완성
# 이름 2개일 경우 shift tab 하면 다음 이름으로 변경됨

# ls: 파일 및 폴더 목록(*cmd에서는 dir)

#cls : 터미널 내용 지우기(화면정리)
a=10
a
a='"안녕하세요!"라고 아빠가 말했다.'
a
a='안녕하세요!'
a
a=[1,2,3]
a
b=[4,5,6]
a+b
c=123
a+c
a='안녕하세요!'
b='LS 빅데이터 스쿨!'
a+' '+b
c=123
# a+c는 에러
d="123"
a+d
a
print(a)

a=10
b=3.3
print("a+b=",a+b)
print("a-b=",a-b)
print("a*b=",a*b)
print("a/b=",a/b)
print("a%b=",a%b)
print("a//b=",a//b)
print("a**b=",a**b)

print(10**3)
print(100//7)
print(100%7)

# shift+Alt+아래화살표: 아래로 복사
# ctrl+Alt+아래화살표: 커서 여러개 (여러 줄 한번에 수정할때)

(a**3)//7
(a**3)//7*2
(a**3)//7*2
(a**3)//7*2
(a**2)//7
(a**)

a=10
b=3.3
a==b
a!=b
a>=b
a<=b
a>b
a<b

# 2에 4승과 12453을 7로 나눈 몫을 더해서 8로 나눴을 떄 나머지
a= ((2**4) + (12453 // 7 )) % 8
a
# 9의 7승을 12로 나누고, 36452를 253로 나눈 나머지에 곱한 수
b= ((9**7) / 12) * (36452 % 253)
b
# 중 큰 것은?
a<b

user_age=14
is_adult=user_age>=18
print("성인입니까?", is_adult)

TRUE = "하이"
true = "안녕"

a = "True"
b = TRUE 
C = true 
d = True

# True, False

a= True
b= False

a and b
a or b

not a


# True : 1
# Flase : 0
True + True # 2
True + False # 1
False + False # 0

# and 연산자 
True and False # False
True and True # True
False and False # False
False and True # False

# or 연산자
True or True # True
True or False # True
False or True # True
False or False # False

# and는 곱셈으로 치환 가능
True * False # 0
True * True # 1
False * False # 0
False * True # 0

# or 연산자
True + True # True
True + False # True
False + True # True
False + False # False

a= False
b= False
a or b

min(a+b,1)

a = 3
a += 10
a
a-=4
a
a %= 3
a

a += 12
a

a**= 2
a
a /= 7
a

str1 = "hello "
str1 + str1 # 'hello hello'
c = str1 * -1 # ''
print(c)
# 정수 : int(eger)
# 실수: float(double)

x = 5
+x # 5
-x # -5
~x # -6
~-x # 4

 # binary
bin(5) # '0b101'
bin(~5) # '-0b110'

bin(-5) # '-0b101'
bin(~-5) # '0b100'
int('0b11111010', 2)

bin(1)
bin(~1)
1
~1
bin(3)
bin(~3)
2
bin(15)
5
~5
~-5
~-3
bin(-3)
bin(~-3)
int('0b11110001',2)
~-5
(-128+64+32+16+8+2)

# 콘솔: !pip install pydataset !는 파이썬 밖에다 해줘~ 라는 뜻
# 터미널: 파이썬 밖이라 !안하고 그냥 깔림

import pydataset

pydataset.data() # 데이터 목록 반환
df = pydataset.data("AirPassengers") # 해당 데이터 반환

import pandas as pd

df = pd.DataFrame({'name': ['김지훈','이유진','박동현','김민지'],
'engilsh': [90,80,60,70],
'math':[50,60,100,20]})
df

# 테스트
