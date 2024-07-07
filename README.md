# study-Python_ML

# -*- coding: utf-8 -*-
"""
step01_Series.py

Series 객체 특징 
 - pandas 1차원(vector) 자료구조 
 - DataFrame의 칼럼 구성요소 
 - 수학/통계 관련 함수 제공 
 - indexing/slicing 기능 
"""

import pandas as pd  # pd.Series() 
from pandas import Series # Series() 


# 1. Series 객체 생성 

# 1) list 이용 
price = pd.Series([3000,2000,1000,5200]) # [list]
print(price)
'''
index values 
0    3000
1    2000
2    1000
3    5200
dtype: int64 -> 자료형(동일) 
'''

# 숫자 색인(10진수)
price[3] # 5200 : indexing 
price[1:3] 
'''
1    2000
2    1000
'''

# 조건 색인 
price[price >= 3000]



# 2) dict 이용 : key -> index, value -> values 
person = Series({'name':'홍길동', 'age':35, 'addr':'서울시'}) 
print(person)
'''
name    홍길동
age      35
addr    서울시
dtype: object -> 자료형(동일) 
'''

# 명칭 색인 
person['name'] # '홍길동'


# 2. indexing/slicing 
ser = Series([4, 4.5, 6, 8, 10.5])  
print(ser)

# list 동일 
ser[:] # 전체 원소 
ser[0] # 1번 원소 
ser[:3] # start~2번 원소 
ser[3:] # 3~end 원소 

#ser[-1] # KeyError: -1


# 3. Series 결합과 NA 처리 
s1 = pd.Series([3000, None, 2500, 2000],
               index = ['a', 'b', 'c', 'd'])

s2 = pd.Series([4000, 2000, 3000, 1500],
               index = ['a', 'c', 'b', 'd'])


# Series 결합(사칙연산)
s3 = s1 + s2 # index 기준 연산 
print(s3)
'''
a    7000.0
b       NaN -> 결측치(Not a Number) 
c    4500.0
d    3500.0
'''
s3.mean() # 5000.0

type(s3) # pandas.core.series.Series
dir(s3)
'''
fillna(값) : 결측치 채우기 
notnull() : 결측치 제외 
isna() or isnull() : 결측치 유무 확인 
'''

# 결측치 유무 확인
s3.isnull() # True 
s3.isnull().sum() # 1 : 결측치 총개수 


# 결측치 처리
result = s3.fillna(s3.mean()) # 평균 대체 
print(result)
'''
a    7000.0
b    5000.0 -> 평균 대체
c    4500.0
d    3500.0
''' 
result2 = s3.fillna(0) # 0으로 대체 
print(result2)
'''
a    7000.0
b       0.0
c    4500.0
d    3500.0
'''

# 결측치 제거 : 결측치 제외 
result3 = s3[s3.notnull()] # s3[pd.notnull(s3)] 
print(result3)


# 4. Series 연산 

# 1) 범위 수정 
print(ser)
ser[1:4] = 5.0


# 2) broadcast 연산 
print(ser * 0.5) # 1d * 0d(상수)  

# 3) 수학/통계 함수 
dir(ser)
ser.mean() # 평균
ser.sum() # 합계
ser.var() #  분산
ser.std() # 표준편차
ser.max() # 최댓값
ser.min() # 최솟값

# 유일값 
ser.unique() # [ 4. ,  5. , 10.5]
# 출현 빈도수 
ser.value_counts()
'''
5.0     3
4.0     1
10.5    1
'''
