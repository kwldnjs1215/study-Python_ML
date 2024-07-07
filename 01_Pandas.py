# -*- coding: utf-8 -*-
"""
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

########################################################################################################################
"""
DataFrame 자료구조 특징 
 - 2차원 행렬구조(DB의 Table 구조와 동일함)
 - 칼럼 단위 상이한 자료형 
"""

import pandas as pd # 별칭 
from pandas import DataFrame 


# 1. DataFrame 객체 생성 

# 1) list와 dict 이용 
names = ['hong', 'lee', 'kim', 'park']
ages = [35, 45, 55, 25]
pays = [250, 350, 450, 250]


# key -> 칼럼명, value -> 칼럼값 
frame = pd.DataFrame({'name':names, 'age': ages, 'pay': pays})
frame
'''
   name  age  pay -> 열이름(칼럼)
0  hong   35  250
1   lee   45  350
2   kim   55  450
3  park   25  250
'''

# 객체 정보 
frame.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3 : 관측치 
Data columns (total 3 columns): 열이름(칼럼)
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   name    4 non-null      object : 문자형 
 1   age     4 non-null      int64  : 정수형 
 2   pay     4 non-null      int64  : 정수형 
dtypes: int64(2), object(1)
'''


# 2) numpy 객체 이용
import numpy as np

data = np.arange(12).reshape(3, 4) # 1d -> 2d
print(data) 

# numpy -> pandas
frame2 = DataFrame(data, columns=['a','b','c','d'])
frame2


# 2. DF 칼럼 참조 
path = r'C:/ITWILL/4_Python_ML/data' # 경로 지정
emp = pd.read_csv(path + "/emp.csv", encoding='utf-8')
print(emp.info())
print(emp)
emp.shape # (5, 3)


# 1) 단일 칼럼
no = emp.No # 방법1
name = emp['Name'] # 방법2(공백이나 콤마(.) 포함)


# 2) 복수 칼럼  
df = emp[['No','Pay']] # 중첩list 
type(df) # pandas.core.frame.DataFrame


# 3. DataFrame 행열 참조 
'''
DF.loc[행label,열label]
DF.iloc[행index,열index]
'''
# 1) loc 속성 : 명칭 기반 
emp.loc[0, 'No':'Pay'] # 1행 전체 
emp.loc[0] # 1행 전체 : 열 생략 가능 
emp.loc[0:2] # 1~3행 전체 
# 숫자 색인 -> 명칭 색인 
emp.loc[1:3,'Name':'Pay'] # box 선택 

# 2) iloc 속성 : 숫자 기반 
emp.iloc[0] # 1행 전체 
emp.iloc[0:2] # 1~2행 전체 
emp.iloc[:,1:] # 2번째 칼럼 이후 연속 칼럼 선택
emp.iloc[1:4, 1:] # box 선택


# 4. subset 만들기 : 기존 DF -> 새로운 DF

# 1) 특정 칼럼 선택 : 칼럼 수가 적은 경우 
subset1 =  emp[['Name', 'Pay']] # 중첩list 
print(subset1)

# 2) 특정 행 제외 
subset2 = emp.drop([1,3]) # 2행,4행 제외 : drop(행index)  
print(subset2)


# 3) 조건식으로 행 선택 : 비교연산자 이용    
subset3 = emp[emp.Pay >= 350] # 급여 350 이하 제외 
print(subset3)


# 논리연산자 이용 : &(and), |(or), ~(not) 
emp[(emp.Pay >= 300) & (emp.Pay <= 400)] # 급여 300 ~ 400   
'''
    No Name  Pay
3  104  유관순  350
4  105  김유신  400
'''

emp[~((emp.Pay >= 300) & (emp.Pay <= 400))] # 급여 300 ~ 400
'''
    No Name  Pay
0  101  홍길동  150
1  102  이순신  450
2  103  강감찬  500
'''


# 4) 칼럼값 이용 : DF.column.isin([값1, 값2]) 
iris = pd.read_csv(path + '/iris.csv') # 붖꽃 datase 
iris.info()
'''
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Sepal.Length  150 non-null    float64 : 꽃받침 길이 
 1   Sepal.Width   150 non-null    float64 : 꽃받침 넓이 
 2   Petal.Length  150 non-null    float64 : 꽃잎 길이 
 3   Petal.Width   150 non-null    float64 : 꽃잎 넓이 
 4   Species       150 non-null    object  : 꽃의 종
'''

#sepal_len = iris.Sepal.Length # 방법1 : AttributeError: 오류 
sepal_len = iris['Sepal.Length']  # 방법2 : 공백 또는 콤마(.) 포함  

iris.Species.value_counts() # 방법1
'''
setosa        50
versicolor    50
virginica     50
'''


print(iris.Species.unique()) # ['setosa' 'versicolor' 'virginica']

subset4 = iris[iris.Species.isin(['setosa', 'virginica'])]
subset4.shape # (100, 5)


# 5) columns 이용 : 칼럼이 많은 경우 칼럼명 이용 
dir(iris) # columns
iris.columns

names = list(iris.columns) # 전체 칼럼명 list 반환 

names # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']

# 독립변수 선택 : 4개 선택 
iris_x = iris[names[:4]] 
# iris[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]

# 종속변수 선택 : 1개 선택 
iris_y = iris[names[-1]]


# 특정 칼럼 제외한 나머지 subset 만들기 
names.remove('Sepal.Width') # Sepal.Width 칼럼 제거  
names # ['Sepal.Length', 'Petal.Length', 'Petal.Width', 'Species']

iris_subset = iris[names]

########################################################################################################################

"""
1. DataFrame의 요약통계량 : 숫자형 변수 
2. 상관계수 & 공분산 : 변수 간의 상관성(크기와 방향) 
"""

import pandas as pd 

path = r'C:\ITWILL\4_Python_ML\data'

product = pd.read_csv(path + '/product.csv')


# DataFrame 정보 보기 
product.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 264 entries, 0 to 263
Data columns (total 3 columns):
'''
    

# 앞부분/뒷부분 관측치 5개(기본값) 보기 
product.head(10)
product.tail(10)

# 1. DataFrame의 요약통계량 
summ = product.describe()
print(summ)
type(summ) # pandas.core.frame.DataFrame

summ.loc['std','a':'c']
'''
a    0.970345
b    0.859657
c    0.828744
'''

# 행/열 통계
product.shape # (264, 3)
product.sum(axis = 0) # 행축(default) : 열단위   
product.sum(axis = 1) # 열축 : 행단위 

# 산포도 : 분산, 표준편차 
product.var() # axis = 0
product.std() # axis = 0

# 빈도수 : 집단변수 
product['a'].value_counts()
'''
3    126
4     64
2     37
1     30
5      7
'''

# 최빈수 
product['a'].mode() # 3

# 유일값 
product['b'].unique()
# [4, 3, 2, 5, 1]


# 2. 상관관계 
cor = product.corr()
print(cor) # 상관계수 행렬 : -1 ~ +1 
'''
          a         b         c
a  1.000000  0.499209  0.467145
b  0.499209  1.000000  0.766853
c  0.467145  0.766853  1.000000
'''

# 공분산 
cov = product.cov()
print(cov) # 공분산 행렬 
'''
          a         b         c
a  0.941569  0.416422  0.375663
b  0.416422  0.739011  0.546333
c  0.375663  0.546333  0.686816
'''

########################################################################################################################

import pandas as pd 
pd.set_option('display.max_columns', 100) # 콘솔에서 보여질 최대 칼럼 개수 

path = r'C:\ITWILL\4_Python_ML\data'

wdbc = pd.read_csv(path + '/wdbc_data.csv')
wdbc.info()
'''
RangeIndex: 569 entries, 0 to 568
Data columns (total 32 columns):
'''
print(wdbc.head())


# 전체 칼럼 가져오기 
cols = list(wdbc.columns)


# 1. DF 병합(merge) : 공통칼럼 기준 
DF1 = wdbc[cols[:16]] # id
DF2 = wdbc[cols[16:]] # id(x) 

# 공통칼럼 추가 : id 
DF2['id'] = wdbc.id

DF1.shape # (569, 16)
DF2.shape # (569, 17)

DF3 = pd.merge(left=DF1, right=DF2, on='id') 
DF3.shape # (569, 32)


# 2. DF 결합(concat)
DF2 = wdbc[cols[16:]]

DF4 = pd.concat(objs=[DF1, DF2], axis = 1) # 열축 기준 결합(cbind)
DF4.shape # (569, 32)


# 3. Inner join과 Outer join 
name = ['hong','lee','park','kim']
age = [35, 20, 33, 50]

df1 = pd.DataFrame(data = {'name':name, 'age':age}, 
                   columns = ['name', 'age'])

name2 = ['hong','lee','kim']
age2 = [35, 20, 50]
pay = [250, 350, 250] # 추가 

df2 = pd.DataFrame(data = {'name':name2, 'age':age2,'pay':pay}, 
                   columns = ['name', 'age', 'pay'])

# Inner join
inner = pd.merge(left=df1, right=df2, on=['name','age'], how='inner')
inner
'''
   name  age  pay
0  hong   35  250
1   lee   20  350
2   kim   50  250
'''

# left Outer join : left 기준 
outer = pd.merge(left=df1, right=df2, on=['name','age'], how='outer')  
outer
'''
   name  age    pay
0  hong   35  250.0
1   lee   20  350.0
2  park   33    NaN
3   kim   50  250.0
'''

########################################################################################################################

import pandas as pd 

path = r'C:\ITWILL\4_Python_ML\data'

buy = pd.read_csv(path + '/buy_data.csv')

print(buy.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22 entries, 0 to 21
Data columns (total 3 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   Date         22 non-null     int64 : 구매일 
 1   Customer_ID  22 non-null     int64 : 고객id
 2   Buy          22 non-null     int64 : 수량 
'''
print(buy)
buy.shape # (22, 3) : 2d 

# 1. 2차원(wide) -> 1차원(long) 구조 변경
buy_long = buy.stack() # 22*3
buy_long.shape # (66,) : 1d 

# 2. 1차원(long) -> 2차원(wide) 구조 변경 
buy_wide = buy_long.unstack() # 복원 기능 
buy_wide.shape # (22, 3) : 2d 

# 3. 전치행렬 : 행축 <-> 열축  
buy_tran = buy.T
buy_tran.shape # (3, 22)


# 4. 중복 행 제거
dir(buy) 
buy.duplicated() # 중복 행 확인 
'''
10     True
16     True
'''

buy2 = buy.drop_duplicates() # 중복 행 제거
buy2.shape # (20, 3)
buy2


# 5. 특정 칼럼을 index 지정 
new_buy = buy.set_index('Date') # 구매날짜 
new_buy.shape # (22, 2)
new_buy

# 날짜 검색 
new_buy.loc[20150101] # 명칭색인 
'''
          Customer_ID  Buy
Date                      
20150101            1    3
20150101            2    4
20150101            2    3
20150101            1    2
'''
new_buy.loc[20150107]
'''
          Customer_ID  Buy
Date                      
20150107            3    4
20150107            5    3
20150107            1    9
20150107            5    7
'''

#new_buy.iloc[20150101] # 오류 : out-of-bounds : 색인 범위 초과 
# 색인 자료형은 int형이지만 동일한 구매날짜를 명칭으로 지정한다.  

# [추가] 주가 dataset 적용 
stock = pd.read_csv(path + '/stock_px.csv')
stock.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2214 entries, 0 to 2213
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Unnamed: 0  2214 non-null   object 
 1   AAPL        2214 non-null   float64
 2   MSFT        2214 non-null   float64
 3   XOM         2214 non-null   float64
 4   SPX         2214 non-null   float64
'''
stock.head() 

# 칼럼명 수정  
stock.columns = ['Date','AAPL','MSFT','XOM','SPX']

# object -> date형 변환 
stock['Date'] = pd.to_datetime(stock.Date) 

stock.info()
# 0   Date    2214 non-null   datetime64[ns]

# Date 칼럼 색인 지정 
new_stock = stock.set_index('Date')
new_stock.head() # 2003-01-02
new_stock.tail() # 2011-10-10

# subset 만들기 
app_ms = new_stock[['AAPL','MSFT']]

# 8년치 주가 변동량 
app_ms.plot() 

# 2011년도 주가 변동량 
app_ms.loc['2011'].plot()

# 2008~2011년도 주가 변동량 
app_ms.loc['2008':'2011'].plot()

# 2011년도 2분기 주가 변동량
app_ms.loc['2011-04':'2011-07'].plot()


########################################################################################################################

import pandas as pd 

path = r'C:\ITWILL\4_Python_ML\data'

# 1. csv file read

# 1) 칼럼명이 없는 경우 
st = pd.read_csv(path + '/student.csv', header=None)
st # 0     1    2   3 -> 기본 칼럼명 

# 칼럼명 수정 
col_names = ['sno','name','height','weight'] # list 
st.columns = col_names # 칼럼 수정 
print(st)


# 2) 칼럼명 특수문자(.) or 공백 
iris = pd.read_csv(path + '/iris.csv')
print(iris.info())

#iris.Sepal.Length # AttributeError
iris['Sepal.Length']

# 점(.) -> 언더바(_) 교체 
iris.columns = iris.columns.str.replace('.','_') # ('old','new')
iris.info() # Sepal_Length
iris.Sepal_Length


# 3) 특수구분자(tab키), 천단위 콤마 
# pd.read_csv('file', delimiter='\t', thousands=',')


# 2. data 처리 : 파생변수 추가 
'''
비만도 지수(bmi) = 몸무게/(키**2)
몸무게 : kg
키 : m
'''

bmi = st.weight / (st.height*0.01)**2
bmi
'''
0    21.224490
1    24.835646
2    20.047446
3    21.604938
'''
    
# bmi 파생변수 추가 : 비율척도(연속형) 
st['bmi'] = bmi
st
'''
label : normal, fat, thin 
normal : bmi : 18 ~ 23
fat : 23 초과
thin : 18 미만  
'''

# label 파생변수 추가 : 명목척도(범주형)
label = [] 

for bmi in  st.bmi : 
    if bmi >= 18 and bmi <= 23 :
        label.append('normal')
    elif bmi > 23 :
        label.append('fat')
    else :
        label.append('thin')

# 파생변수 추가 
st['label'] = label

print(st)

dir(st)
'''
to_csv : csv file 저장 
to_excel : excel file 저장 
to_json : json file 저장 
'''

# 3. csv file 저장 
st.to_csv(path + '/st_info.csv', index = None, encoding='utf-8')
# index = None : 행이름 저장 안함 

########################################################################################################################

"""
1. 범주형 변수 기준 subset 만들기
2. 범주형 변수 기준 group & 통계량
3. apply() 함수 : DataFrame(2D) 객체에 함수 적용 
4. map() 함수 : Series(1D) 객체에 함수 적용 
"""

import pandas as pd 

 
path = r'C:\ITWILL\4_Python_ML\data'

# dataset load & 변수 확인
wine = pd.read_csv(path  + '/winequality-both.csv')
print(wine.info())
'''
RangeIndex: 6497 entries, 0 to 6496
Data columns (total 13 columns):
0   type                  6497 non-null   object : 와인유형(범주형변수)    
 :
12  quality               6497 non-null   int64 : 와인품질(이산형변수) 
'''    

# 칼럼 공백 -> '_' 교체 
wine.columns = wine.columns.str.replace(' ', '_')
wine.head()
print(wine.info())


# 5개 변수 선택 : subset 만들기 
wine_df = wine.iloc[:, [0,1,4,11,12]] # 위치 기반
print(wine_df.info()) 

# 특정 칼럼명 수정 
columns = {'fixed_acidity':'acidity', 'residual_sugar':'sugar'} # {'old','new'} 
wine_df = wine_df.rename(columns = columns) 
wine_df.info()
'''
DF.columns = [칼럼명] : 전체 칼럼 수정 
DF.rename(columns = {'기존칼럼명':'새칼럼명'})
'''    

# 범주형 변수 확인 : 와인유형   
print(wine_df.type.unique()) # ['red' 'white']
print(wine_df.type.nunique()) # 2
wine_df.type.value_counts()
'''
white    4898
red      1599
'''

# 이산형 변수 확인 : 와인 품질    
print(wine_df.quality.unique()) # [5 6 7 4 8 3 9]
print(wine_df.quality.value_counts())
'''
6    2836
5    2138
7    1079
4     216
8     193
3      30
9       5
'''

# 1. 범주형 변수 기준 subset 만들기 

# 1) 1개 집단 기준  
red_wine = wine_df[wine['type']=='red']  # DF[DF.칼럼 조건식]
red_wine.shape # (1599, 5)

white_wine = wine_df[wine['type'] == 'white']
white_wine.shape # (4898, 5)


# 2) 2개 이상 집단 기준 : type : {red(o), blue(x), white(o)}
two_wine_type = wine_df[wine_df['type'].isin(['red','white'])] 


# 3) 범주형 변수 기준 특정 칼럼 선택 : 1차원 구조
red_wine_quality = wine.loc[wine['type']=='red', 'quality']  
red_wine_quality.shape # (1599,)

white_wine_quality = wine.loc[wine['type']=='white', 'quality'] 
white_wine_quality.shape # (4898,)


# 2. 범주형 변수 기준 group & 통계량

# 1) 범주형변수 1개 이용 그룹화
# 형식) DF.groupby('범주형변수') 

type_group = wine_df.groupby('type')
print(type_group) # DataFrameGroupBy object 정보 

# 각 집단별 빈도수 
type_group.size()  
'''
red      1599 5
white    4898 5
'''

# 그룹객체에서 그룹 추출 
red_df = type_group.get_group('red')
white_df = type_group.get_group('white')
red_df.shape # (1599, 5)
white_df.shape # (4898, 5)
    
# 그룹별 통계량 : 연산과정 ppt.52 참고 
print(type_group.sum()) 
'''
        acidity     sugar   alcohol  quality
type                                        
red    13303.10   4059.55  16666.35     9012
white  33574.75  31305.15  51498.88    28790
'''

print(type_group.mean())
'''
        acidity     sugar    alcohol   quality
type                                          
red    8.319637  2.538806  10.422983  5.636023
white  6.854788  6.391415  10.514267  5.877909
'''


# 2) 범주형 변수 2개 이용 : 나머지 변수(3개)가 그룹 대상 
# DF.groupby(['범주형변수1', '범주형변수2'])
wine_group = wine_df.groupby(['type','quality']) # 2개 x 7개 = 최대 14  

# 각 집단별 빈도수
wine_group.size()
'''
1차     2차 
type   quality
red    3            10
       4            53
       5           681
       6           638
       7           199
       8            18
white  3            20
       4           163
       5          1457
       6          2198
       7           880
       8           175
       9             5
'''
       
# 그룹 통계 시각화 
grp_mean = wine_group.mean()
grp_mean

grp_mean.plot(kind='bar')


# 3. apply() 함수 : DataFrame(2D) 객체에 외부함수 적용

# 1) 사용자 함수 : 0 ~ 1 사이 정규화 
def normal_df(x):
    nor = ( x - min(x) ) / ( max(x) - min(x) )
    return nor


# 2) 2차원 data 준비 : wine 데이터 적용 
wine_x = wine_df.iloc[:, 1:] # 숫자변수만 선택 
wine_x.shape # (6497, 4)

wine_x.describe()

# 3) apply 함수 적용 : 열(칼럼) 단위로 실인수 전달
# 형식) DF.apply(내장함수/사용자함수)   
wine_nor = wine_x.apply(normal_df) 
wine_nor
print(wine_nor.describe()) # 정규화 확인 


# 4. map() 함수 : Series(1D) 객체에 함수 적용   

# 1) 인코딩 함수 
def encoding_df(x):
    encoding = {'red':[1,0], 'white':[0,1]}
    return encoding[x]

# 2) 1차원 data 준비 
wine_type = wine_df['type']
wine_type.shape # (6497,)
type(wine_type) # pandas.core.series.Series

# 3) map 함수 적용 
# 형식) Series.map(내장함수/사용자함수)
label = wine_type.map(encoding_df)
label

# lambda 이용 : 한 줄 함수 적용 
encoding = {'red':[1,0], 'white':[0,1]} # dict : mapping table 

wine_df['label'] = wine_df['type'].map(lambda x : encoding[x])

wine_df.head()
wine_df.tail()

########################################################################################################################

"""
피벗테이블(pivot table) 
  - DF 객체를 대상으로 행과 열 그리고 교차 셀에 표시될 칼럼을 지정하여 만들어진 테이블 
   형식) pivot_table(DF, values='교차셀 칼럼',
                index = '행 칼럼', columns = '열 칼럼'
                ,aggFunc = '교차셀에 적용될 함수')  
"""

import pandas as pd 

path = r'C:\ITWILL\4_Python_ML\data'

# csv file 가져오기 
pivot_data = pd.read_csv(path + '/pivot_data.csv')
pivot_data.info()
'''
 0   year     8 non-null      int64  : 년도 
 1   quarter  8 non-null      object : 분기 
 2   size     8 non-null      object : 매출규모
 3   price    8 non-null      int64  : 매출액 
'''
# 1. 핏벗테이블 작성
ptable = pd.pivot_table(data=pivot_data, 
               values='price', 
               index=['year','quarter'], 
               columns='size', aggfunc='sum')
 
print(ptable)


# 2. 핏벗테이블 시각화 : 누적형 가로막대 
ptable.plot(kind='barh', stacked=True)















