#데이터 타입
x=15
print(x, "는", type(x), "형식입니다.", sep='*')


y="hi"
print(y, type(y))

z="""This is
a multi-line
string
"""
print(z, type(z))

greeting = "안녕"+" "+"파이썬!"
print("결합된 문자열:", greeting)

laugh = "하"*3
print("반복된 문자열:", laugh)

# 리스트
fruit = ["apple","banana","cherry"]
print(type(fruit))

number = [1,2,3,4,5]
print(type(number))

mixed_list = [1,"hello",[1,2,3]]
print(type(mixed_list))

a=(10,20,30) # a=1,2,3이라고 치면 자동으로 튜플로 저장됨
# 튜플이 리스트보다 가볍다

a[0]
a[1]
a[2]

# deepcopy
a=[1,2,3]
id(a)
b=a[:]
b
id(b)
a[1] = 4
a
b

a
b=(42)
b
type(b)
b=10
b
b=(42,)
b
type(b)
b=10
b

a=(10,20,30,40,50,60,70)
a[3:] # 해당 인덱스부터 끝까지
a[:3] # 처음부터 해당 인덱스 전까지
a[1:3]


b=[10,20,30,40,50,60,70]
b[3:6]
b[3:]
b[:4]

a[1]
b[1]
a[1]=25
b[1]=25
a
b


def min_max(numbers):
  return min(numbers), max(numbers) #튜플로 반환됨

a=[1,2,3,4,5]
result = min_max(a) # (1,5)
result[0] = 4
type(result)
print("minimun and maximun:", result)

#딕셔너리
person = {
  'name':'John',
  'age':30,
  'city':'New York'
}
issac= {
  'name':'이삭',
  "나이":(39,30),
  "사는곳":["미국","한국"]
}
print("Person:", person)
print("Issac:", issac)

issac_age=issac.get('나이')
issac_age[0]

fruits= {'apple','banana','cherry','apple'}
print(fruits)
type(fruits)

# 빈 집합 생성
empty_set = set()
print(empty_set)
empty_set.add('apple')
empty_set.add('banana')
empty_set
empty_set.add('apple')
empty_set
empty_set.remove('banana')
empty_set
empty_set.discard('banana')
empty_set

fruits={'cherry', 'apple', 'banana'}
other_fruits={'berry', 'cherry'}
union_fruits = fruits.union(other_fruits)
intersection_fruits=fruits.intersection(other_fruits)
union_fruits
intersection_fruits

type(True)
print(True)
print(True+True)

age = 10
is_greater = age>5
print(is_greater)

# 조건문
a=3
if(a>5):
  print("a는 2와 같습니다")
else:
  print("a는 2와 같지 않습니다.")
  
# 숫자열을 문자열형으로 변환
num = 123
str_num = str(num)
str_num
float_num = float(str_num)
float_num
type(float_num)
print(str_num, type(str_num))

lst = [1,2,3]
typ = tuple(lst)
typ
type(typ)

set_example = {'a','b','c'}
dict_from_set = {key:True for key in set_example}
print(dict_from_set)

set_from_dict = set(dict_from_set.values())
set_from_dict

# 교재 63페이지
import seaborn as sns
import matplotlib.pyplot as plt
# !pip install seaborn

var = ['a','a','b','c']
var

sns.countplot(x=var)
plt.show()
plt.clf()

df = sns.load_dataset("titanic")
sns.countplot(data=df, x="sex", hue="sex")
plt.show()
plt.clf()

sns.countplot(data=df, x="class")
plt.show()
plt.clf()

sns.countplot(data=df, x="class",hue="alive")
plt.show()
plt.clf()

?sns.countplot
sns.countplot(data=df, y="class",hue="alive")
plt.show()
plt.clf()

# !pip install scikit-learn

import sklearn.metrics
sklearn.metrics.accuracy_score()

from sklearn import metrics
metrics.accuracy_score()

from sklearn.metrics as met
met.accuracy_score()
