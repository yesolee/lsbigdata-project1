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

result = matrix + vector
print("브로드캐스팅 결과:\n", result)
 
