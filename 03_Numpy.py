"""
Numpy 패키지 
  - 수치 과학용 데이터 처리 목적으로 사용 
  - 선형대수(벡터, 행렬) 연산 관련 함수 제공 
  - N차원 배열, 선형대수 연산, 고속 연산  
  - 수학/통계 함수 제공 
  - indexing/slicing   
  - broadcast 연산
"""

import numpy as np # 별칭 


# 1. list 배열 vs numpy 다차원 배열 

# 1) list 배열
lst = [1, 2, 3, 3.5] # 정수와 실수 자료형
print(lst) 
print(lst * 3 ) # 3번 반복
sum(lst) # 외부 함수 

# 2) numpy 다차원 배열 
arr = np.array(lst) # array([list])  
print(arr) 
print(arr * 0.5) # broadcast 연산 
arr.sum() # 자체 제공 


# 2. array() : 다차원 배열 생성 

lst1 = [3, 5.2, 4, 7]
print(lst1) # 단일 리스트 배열 

arr1d = np.array(lst1) # array(단일list)
print(arr1d.shape) # (4,) : 자료구조 확인

print('평균 =', arr1d.mean()) 
print('분산=', arr1d.var())
print('표준편차=', arr1d.std()) 


# 3. broadcast 연산 
# - 작은 차원이 큰 차원으로 늘어난 후 연산 

# scala(0) vs vector(1)
print(0.5 * arr1d)


'''
broadcast 연산 예 
모집단 분산 = sum((x-mu)**2) / n
표본 분산 = sum((x-x_bar)**2) / n-1
'''
x = arr1d # 객체 복제 
x # [3. , 5.2, 4. , 7. ]

mu = x.mean() # 모평균  
var = sum((x - mu)**2) / len(x)


# 4. zeros or ones 
zerr = np.zeros( (3, 10) ) # 3행10열 
print(zerr) 

onearr = np.ones( (3, 10) ) # 3행10열 
print(onearr)


# 5. arange : range 유사함
range(1, 11) # 1 ~ 10
#range(0.5, 10.5) # TypeError
 
arr = np.arange(-1.2, 5.5) # float 사용 가능, 배열 객체  
print(arr) # [-1.2 -0.2  0.8  1.8  2.8  3.8  4.8]

# ex) x의 수열에 대한 2차 방정식 
x = np.arange(-1.0, 2, 0.1) # (start, stop, step)

y = x**2 + 2*x + 3 # f(x) 함수 
print(y)

import matplotlib.pyplot as plt 

plt.plot(x, y)
plt.show()

##################################################################################################################

"""
indexing/slicing 
 - 1차원 indexing 
 - 2,3차원 indexing 
 - boolean indexing  
"""

import numpy as np

# 1. 색인(indexing) 

# 1) list 배열 색인
ldata = [0,1,2,3,4,5]
print(ldata[:]) # 전체 원소 
print(ldata[3]) # 특정 원소 1개 
print(ldata[:3]) # 범위 선택 (0~n-1)
print(ldata[-1]) # 오른쪽 기준(-)

# 2) numpy 다차원 배열 색인 : list 동일 
arr = np.arange(10) # 0~9
print(arr[:])
print(arr[3])
print(arr[:3])
print(arr[-1])


# 2. slicing : 특정 부분을 잘라서 new object
arr = np.arange(10) # 0~9
arr # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 주소 복사 
arr_obj = arr[1:4] # 주소 반환 
print(arr_obj) # [1 2 3]

arr_obj[:] = 100 # 전체 수정(o)
print(arr_obj) # [100 100 100]

print(arr) # 원본 변경 

# 내용 복사 
arr_obj2 = arr[1:4].copy()

arr_obj2[:] = 200 # 전체 수정 

print(arr) # 원본 변경(x)


# 3. 고차원 색인(indexing) : 2차원 이상 

# 1) 2차원 indexing 
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 중첩list
print(arr2d)
'''
[[1 2 3]  0
 [4 5 6]  1
 [7 8 9]] 2
'''

# 행 index(기본)
print(arr2d[0, :]) # 1행 전체 
print(arr2d[0]) # 1행 전체
print(arr2d[1:,1:])
print(arr2d[::2]) # 홀수행 선택 : [start:stop:step] 

# 비연속 행렬
print(arr2d[[0,2]])
print(arr2d[[0,2], [0,2]]) # 1행1열, 3행3열 


# 2) 3차원 indexing 
arr3d = np.array([[[1,2,3],[4,5,6]], [[7,8,9], [10,11,12]]])
print(arr3d)
'''
[[[ 1  2  3]
  [ 4  5  6]]

 [[ 7  8  9]
  [10 11 12]]]
'''

print(arr3d.shape) # (2, 2, 3) - (면,행,열)

# 면 index(기본)
print(arr3d[0]) # 1면 전체 
print(arr3d[0, 1])  # 1면의 1행 전체 : [4 5 6]
print(arr3d[0, 1, 1:])  # 1면 1행 2~3열 : [5 6]


# 4. 조건식 색인(boolean index)
dataset = np.random.randn(3, 4) # 12개 
print(dataset)
'''
[[-1.21772518  0.55944886 -0.36250449 -0.45470372]
 [-0.38638528 -1.59064206 -0.3654452   2.16059475]
 [ 1.32900456 -0.40916446 -0.58419923  0.11281693]]
'''

# 0.7 이상 경우 
print(dataset[dataset >= 0.7]) # 비교연산자 
# [2.16059475 1.32900456]

# 논리연산자
dir(np.logical_and)
'''
np.logical_and() : 논리곱
np.logical_not() : 부정 
np.logical_or() : 논리합
np.logical_xor() : 배타적 논리합 
''' 
result = dataset[np.logical_and(dataset >= 0.1, dataset <= 1.5)]
print('0.1 ~ 1.5 : 범위 조건식')
print(result) # [0.55944886 1.32900456 0.11281693]

# 기호 사용 : &, ~, |
result = dataset[(dataset >= 0.1) & (dataset <= 1.5)]
print(result) # [0.55944886 1.32900456 0.11281693]

##################################################################################################################

'''
범용 함수(universal function)
  - 다차원 배열의 원소를 대상으로 수학/통계 등의 연산을 수행하는 함수
'''
import numpy as np # 별칭 


# 1. numpy 제공 함수 
data = np.random.randn(5) # 1차원 난수 배열   

print(data) # 1차원 난수 
print(np.abs(data)) # 절대값
print(np.sqrt(data)) # 제곱근 : nan
print(np.square(data)) # 제곱 
print(np.sign(data)) # 부호 
print(np.var(data)) # 분산
print(np.std(data)) # 표준편차

                  
data2 = np.array([1, 2.5, 3.36, 4.6])

# 로그 함수 : 완만한 변화 
-np.log(0.5) # -0.6931471805599453 -> 0.6931471805599453

np.log(data2) # 밑수 e
# [0.        , 0.91629073, 1.19392247, 1.5260563 ]

# 지수 함수 : 급격한 변화 
np.exp(data2)
# [ 2.71828183, 12.18249396, 27.11263892, 99.48431564]

# 반올림 함수 
np.ceil(data2) # [1., 3., 4., 5.] - 큰 정수 올림 
np.rint(data2) # [1., 2., 3., 5.] - 가장 가까운 정수 올림 
np.round(data2, 0) # [1., 2., 3., 5.] - 자릿수 지정 

# 결측치 처리  
data2 = np.array([1, 2.5, 3.3, 4.6, np.nan])
np.isnan(data2) # True 
# [False, False, False, False,  True]

# 결측치 제외 : 조건식 이용 
data2[np.logical_not(np.isnan(data2))] # [1. , 2.5, 3.3, 4.6]

data2[~(np.isnan(data2))] # # [1. , 2.5, 3.3, 4.6]


# 2. numpy 객체 메서드 
data2 = np.random.randn(3, 4) # 2차원 난수 배열
print(data2)
print(data2.shape) # (3, 4)

dir(data2)

print('합계=', data2.sum()) # 합계
print('평균=', data2.mean()) # 평균
print('표준편차=', data2.std()) # 표준편차
print('최댓값=', data2.max()) # 최댓값
print('최솟값=', data2.min()) # 최솟값


# 3. axis 속성 
print(data2.sum(axis=0)) # 행 축 : 열 단위 합계 
print(data2.sum(axis=1)) # 열 축 : 행 단위 합계 
print('전체 원소 합계 :', data2.sum()) # 축 생략 : 전체 원소 합계 

##################################################################################################################

"""
random 모듈
 - 난수 생성 함수 제공 
 - 확률 실험에서 표본 추출 
 - np.random.*
"""

import numpy as np # 난수 모듈  
import matplotlib.pyplot as plt # 그래프
import pandas as pd # csv file 


# 1. 난수 실수와 정수  

# 1) 난수 실수 : [0, 1) -> 0 <= r < 1
data = np.random.rand(5, 3) # (행, 열)
print(data)

# 차원 모양
print(data.shape) # (5, 3) 

# 난수 통계
print(data.min()) # 0.08132439151228676
print(data.max()) # 0.9462959398267136
print(data.mean()) # 0.63050758946005

# 2) 난수 정수 : [a, b) -> a <= r < b  
data = np.random.randint(165, 175, size=50) # (행, 열)
print(data)

# 차원 모양
print(data.shape) # (50,)

# 난수 통계
print(data.min()) # 165
print(data.max()) # 174
print(data.mean()) # 169.78


dir(np.random)
'''
binomial : 이항분포(이산확률분포) 
poisson : 포아송분포(이산확률분포)
gamma : 감마분포(연속확률분포) 
normal : 정규분포(연속확률분포)  
randn : 표준정규분포(연속확률분포)  
uniform : 균등분포(연속확률분포)
seed : 시드값 
'''

# 2. 이항분포 
np.random.seed(12) # 시드값 
np.random.binomial(n=1, p=0.5, size=10) # 베르누이분포  
# [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
'''
n : 독립시행(1회) = 베르누이시행 
p : 성공확률 
size : 반복횟수(표본수) 
'''


# 3. 정규분포 : N(mu, sigma^2)
height = np.random.normal(173, 5, 2000) # (mu, sigma, size)
print(height) # (2000,)

height2 = np.random.normal(173, 5, (500, 4))
print(height2) 
height2.shape # (500, 4)


# 난수 통계
print(height.mean()) # 172.8258520054461
print(height2.mean()) # 173.00624682142694

# 정규분포 시각화 
plt.hist(height, bins=100, density=True, histtype='step')
plt.show()


# 4. 표준정규분포 : N(mu=0, sigma=1)
standNormal = np.random.randn(500, 3) # (shape)
standNormal

print(standNormal.mean()) # -0.02068273009795626
print(standNormal.std()) # 0.9966609775470275

# normal 함수 이용 
standNormal2 = np.random.normal(0, 1, (500, 3)) # (mu, sigma, shape)
print(standNormal2.mean())


# 정규분포 시각화 
plt.hist(standNormal[:,0], 
         bins=100, density=True, histtype='step', label='col1')
plt.hist(standNormal[:,1], 
         bins=100, density=True, histtype='step', label='col2')
plt.hist(standNormal[:,2], 
         bins=100, density=True, histtype='step',label='col3')
plt.legend(loc='best')
plt.show()


# 5. 균등분포 
uniform = np.random.uniform(10, 100, 1000) # (low, high, size)
plt.hist(uniform, bins=15, density=True)
plt.show()



# 6. DataFrame sampling

## csv file 가져오기
path = r'C:\ITWILL\4_Python_ML\data'
wdbc = pd.read_csv(path + '/wdbc_data.csv')
print(wdbc.info())


# 1) seed값 적용 
np.random.seed(123)

# 2) pandas sample() 이용  
wdbc_df = wdbc.sample(400)
print(wdbc_df.shape) #  (400, 32)
print(wdbc_df.head())

# 3) training vs test sampling
idx = np.random.choice(a=len(wdbc), size=int(len(wdbc) * 0.7), replace = False)
'''
a = 전체 대상(569) 
size = 표본크기(398) 
replace = 복원/비복원 
'''

# training dataset : 70%
train_set = wdbc.iloc[idx]
train_set.shape # (398, 32)

569-398 # 171

# testing dataset : 30%
test_idx = [i for i in range(len(wdbc)) if i not in idx]

test_set = wdbc.iloc[test_idx]
test_set.shape # (171, 32)

##################################################################################################################

"""
reshape : 모양 변경 
 - 1차원 -> 2차원 
 - 2차원 -> 다른 형태의 2차원  
T : 전치행렬 
swapaxis : 축 변경 
transpose : 축 번호 순서로 구조 변경 
"""

import numpy as np

# 1. 모양변경(reshape)
lst = list(range(1, 13)) # 1차원 배열
 
arr2d = np.array(lst).reshape(3, 4) # 모양변경
print(arr2d)
# 주의 : 길이(size) 변경 불가 


# 2. 전치행렬
print(arr2d.T)
print(arr2d.T.shape) # (4, 3)

# 3. swapaxes : 축 변경 
print('swapaxes')
print(arr2d.swapaxes(0, 1)) 


# 4. transpose
'''
2차원 : 전치행렬
3차원 : 축 순서(0,1,2)를 이용하여 자료 구조 변경 
'''
arr3d = np.arange(1, 25).reshape(4, 2, 3)#(4면2행3열)
print(arr3d)
print(arr3d.shape) # (4, 2, 3)

# default : (면,행,열) -> (열,행,면)
arr3d_def = arr3d.transpose() # (2,1,0) 
print(arr3d_def.shape) # (3, 2, 4)
print(arr3d_def)

# user : (면,행,열) -> (열,면,행)
arr3d_user = arr3d.transpose(2,0,1)
arr3d_user.shape # (3, 4, 2)
print(arr3d_user)

##################################################################################################################

import matplotlib.pyplot as plt  
from sklearn.datasets import load_digits 


# 1. image shape & reshape

# 1) dataset load 
digits = load_digits() # 머신러닝 모델에서 사용되는 데이터셋 
dir(digits) # data, target 
'''
입력변수(X) : 숫자(0~9) 필기체의 흑백 이미지 
출력변수(y) : 10진 정수
'''
X = digits.data # 입력변수(X) 추출 
y = digits.target # 출력변수(y) 추출 
X.shape # (1797, 64) : (size, pixel)

X[0].shape # (64,)

# 2) image reshape 
first_img = X[0].reshape(8,8) # 모양변경 : 2d(h,w) 
first_img.shape # (8, 8)

# 3) image show 
plt.imshow(X=first_img, cmap='gray')
plt.show()

# 첫번째 이미지 정답 
y[0] # 0

# 마지막 이미지 
last_img = X[-1].reshape(8,8) # 모양변경 : 2d(h,w)

plt.imshow(X=last_img, cmap='gray')
plt.show()

y[-1] # 8


# 2. image file read & show
import matplotlib.image as img # 이미지 읽기 

# image file path 
path = r"C:\ITWILL\4_Python_ML\workspace\chap03_Numpy" # 이미지 경로 

# 1) image 읽기 
img_arr = img.imread(path + "/data/test1.jpg")
type(img_arr) # numpy.ndarray


img_arr.shape #  (360, 540, 3) : (h, w, color)

# 2) image 출력 
plt.imshow(img_arr)
plt.show()

##################################################################################################################

"""
reshape : 모양변경(크기 변경 불가)
resize : 크기변경 -> 이미지 규격화(모양/크기)
"""

import numpy as np 
import matplotlib.pyplot as plt 

from PIL import Image # PIL(Python Image Lib) : open(), resize()


### 1. image resize
path = r"C:\ITWILL\4_Python_ML\workspace\chap03_Numpy" # 이미지 경로 

# 1) image read 
img = Image.open(path + "/data/test1.jpg") 
type(img) # PIL.JpegImagePlugin.JpegImageFile

np.shape(img) # 원본이미지 모양 : (360, 540, 3) 

# 2) image resize 
img_re = img.resize( (150, 120) ) # (가로, 세로)
np.shape(img_re) # (120, 150, 3)
plt.imshow(img_re)
plt.show()


### 2. 폴더 전체 이미지 resize 
from glob import glob # 파일 검색 패턴 사용(문자열 경로, * 사용) 

img_resize = [] 

for file in glob(path + '/data/*.jpg'): # jpg 파일 검색    
    #print(file) # jpg 파일 경로
    img = Image.open(file) # image read 
    
    img_re = img.resize( (150, 120) ) # image resize
        
    img_resize.append(np.array(img_re)) # numpy array 

print(img_resize)

# list -> array 
img_resize_arr = np.array(img_resize) 

img_resize_arr.shape # (3, 120, 150, 3) : 4d(size, h, w, c)

for i in range(3) :
    plt.imshow(X=img_resize_arr[i])
    plt.show()

##################################################################################################################

"""
 선형대수 관련 함수 
  - 수학의 한 분야 
  - 벡터 또는 행렬을 대상으로 한 연산
  np.linalg.*
"""

import numpy as np 

# 1. 선형대수 관련 함수
 
# 1) 단위행렬 : 대각원소가 1이고, 나머지는 모두 0인 n차 정방행렬
eye_mat = np.eye(3) # eye(차원수)
print(eye_mat)


# 정방행렬 x 
x = np.arange(1,10).reshape(3,3)
print(x)
x.shape # (3, 3)
'''
차원의 의미 
차원 : 독립적인 축이나 방향의 수 
행렬에서 행축의 차원은 데이터의 수(관측치)
행렬에서 열축의 차원은 특징의 수(독립변수) 
'''

# 2) 대각성분 추출 
diag_vec = np.diag(x)
print(diag_vec) # 대각성분 : [1 5 9]
# 용도 : 분류정확도 계산
 

# 3) 대각합 : 정방행렬의 대각에 위치한 원소들의 합 
trace_scala = np.trace(x) 
print(trace_scala) # 15


# 4) 대각행렬 : 대각성분 이외의 모든 성분이 모두 '0'인 n차 정방행렬
diag_mat = eye_mat * diag_vec # 단위행렬 * 대각성분 
print(diag_mat)
'''
[[1. 0. 0.]
 [0. 5. 0.]
 [0. 0. 9.]]
'''

dir(np.linalg)
'''
det() : 행렬식 
eig() : 고유값, 고유벡터(차원축소) 
inv() : 역행렬 
dot() : 행렬곱
norm : 노름(벡터 크기)
solve() : x, y 해 구하기 
svd() : 특이값으로 행렬 분해 
'''


# 5) 행렬식(determinant) : 대각원소의 곱과 차 연산으로 scala 반환 
'''
행렬식 용도 : 역행렬 존재 여부, 행렬식이 0이 아닌 경우 벡터들이 선형적으로 독립 
D(A) = ad - bc = 0 이면 역행렬 존재(x) 
'''

x = np.array([[3,4], [1,2]])
print(x)
'''
  x1 x2 -> 선형적으로 독립
[[3 4]
 [1 2]]
'''

det = np.linalg.det(x)
print(det) # 2.0000000000000004


################################
## 역행렬이 존재하지 않은 경우 
################################

# 1) 행렬 X  
x2 = np.array([[3,0], [1,0]])
print(x2)
'''
 x1 x2 -> 선형적으로 종속 
[[3 0]
 [1 0]]
'''

# 2) 행렬식 
np.linalg.det(x2) # 0.0 : 행렬식 결과 = 0

# 3) 역행렬 : 특이 행렬 - 역행렬 존재 안함(error) 
np.linalg.inv(x2) # LinAlgError: Singular matrix


# 6) 역행렬(inverse matrix) : 행렬식의 역수와 정방행렬 곱 
'''
역행렬 용도 : 회귀 분석에서 최소 제곱 추정
    회귀방정식 :  Y = X * a(회귀계수:가중치)
    최소 제곱 추정 : 손실를 최소화하는 회귀계수를 찾는 역할      
'''
print(x)
'''
[[3 4]
 [1 2]]

1/2 * [[2 -4]
       [-1 3]]
'''

inv_mat = np.linalg.inv(x)
print(inv_mat)
'''
[[ 1.  -2. ]
 [-0.5  1.5]]
'''


################################
## 회귀 분석에서 최소 제곱 추정
################################

# 1) 데이터 생성
X = np.array([[1, 1], [1, 2], [1, 3]]) # 독립변수 
Y = np.array([2, 3, 4]) # 종속변수
print(X)
'''
  x1 x2    Y
[[1 1]  -> 2
 [1 2]  -> 3
 [1 3]] -> 4
'''

# 2) 역행렬
XtX_inv = np.linalg.inv(X.T.dot(X)) # 역행렬 
XtX_inv

# 3) 행렬곱 
XtY = X.T.dot(Y) # array([ 9, 20])


# 4) 최소 제곱 추정 계산
beta_hat = XtX_inv.dot(XtY)  

print("최소 제곱 추정값 :", beta_hat) # 최소 제곱 추정값 : [1. 1.]
# 최소 제곱 추정값 : [1. 1.] -> 회귀계수(가중치)

# 회귀방정식 :  Y = X * a
'''
  x1 x2    Y
[[1 1]  -> 2
 [1 2]  -> 3
 [1 3]] -> 4
'''

# y_pred = (X1 * a1) + (X2 * a2)
y_pred = (1 * 1) + (1 * 1) # 2
y_pred = (1 * 1) + (2 * 1) # 3 
y_pred = (1 * 1) + (3 * 1) # 4 


# 독립변수 행렬 
X.shape # (3, 2)

# 회귀계수(기울기=가중치) 행렬 
a = np.array([[1], [1]])
a.shape # (2, 1)

# 회귀방정식
y_pred = np.dot(X, a) # 행렬곱
y_pred = X.dot(a) # 행렬곱 : 행렬을 곱하고 더해서 상수 반환 
print(y_pred)
'''
[[2]
 [3]
 [4]]
''' 

##################################################################################################################

"""
1. 행렬곱
2. 연립방식의 해 
"""

import numpy as np 

# 1. 행렬곱 : 행렬 vs 행렬 곱셈 연산
'''
  - 행렬의 선형변환(선형결합) : 차원축소 
  - 회귀방정식 : X변수와 기울기(a) 곱셈
  - ANN : X변수와 가중치(w) 곱셈
'''
  

A = np.array([1, 2, 3]).reshape(3,1) # 행렬A
B = np.array([2, 3, 4]).reshape(3,1) # 행렬B

A.shape # (3, 1)
B.shape # (3, 1)
print(B)
#np.dot(A, B) # ValueError: shapes (3,1) and (3,1) not aligned: 1 (dim 1) != 3 (dim 0)
'''
행렬곱 전제조건 
1. A,B 모두 행렬 
2. A의 열의차원 = B의 행의 차원 
'''

# 1) 행렬내적 : 상수 반환  
'''
[[1, 2, 3]]

[[2]
 [3]
 [4]]
'''
dot1 = A.T.dot(B) # [[20]]


# 2) 행렬외적 : 행렬 반환
A.shape # (3, 1)
B.T.shape # (1, 3)
   
dot2 = A.dot(B.T) 
print(dot2)
'''
[[ 2  3  4]
 [ 4  6  8]
 [ 6  9 12]]
'''

# 2. 연립방정식 해(Solve a linear matrix equation): np.linalg.solve(a, b)
'''
연립방정식 : 2개 이상의 방정식을 묶어놓은 것
다음과 같은 연립방정식의 해(x, y) 구하기 
3*x + 2*y = 53
-4*x + 3*y = -35
'''

a = np.array([[3, 2], [-4, 3]]) # 입력 
b = np.array([53, -35]) # 정답 

x, y = np.linalg.solve(a, b)
print(x, y)
'''
13.470588235294118 6.294117647058823
'''

3*x + 2*y # 53.0
-4*x + 3*y # -35.0













    
    
 


































































