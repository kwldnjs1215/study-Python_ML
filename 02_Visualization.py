"""
 - 기본 그래프 그리기 
"""

import matplotlib.pyplot as plt # 시각화 
import random # 난수 생성 


# 차트에서 한글과 음수 부호 지원 
plt.rcParams['font.family'] = 'Malgun Gothic' # 사용할 글꼴 
plt.rcParams['axes.unicode_minus'] = False # 음수 부호 지원


# 1. 그래프 자료 생성 
data = range(-3, 7) # (start, stop-1)
print(data) # [-3, ... ,6]
len(data) # 10


# 2. 기본 그래프 
help(plt.plot)
'''
plot(x, y)        # plot x and y using default line style and color
plot(x, y, 'bo')  # plot x and y using blue circle markers
plot(y)           # plot y using x as index array 0..N-1
plot(y, 'r+')     # ditto, but with red plusses
'''

plt.plot(data) # 선색 : 파랑, 스타일 : 실선 
plt.title('선 색 : 파랑, 선 스타일 : 실선 ')
plt.show() # y축=data, x축=index


# 3. 색상 : 빨강, 선스타일(+)
plt.plot(data, 'r+') # y축=data, x축=index
plt.title('선 색 : 빨강, 선 스타일 : +')
plt.show()


# 4. x,y축 선스타일과 색상 & 마커(circle marker)  
data2 = [random.gauss(0, 1) for i in range(10)]  

plt.plot(data, data2, 'ro') # (x=data, y=data2)
plt.show()

###########################################################################################################

"""
- 이산형 변수 시각화 : 막대 그래프, 원 그래프 
- 이산형 변수 : 셀수 있는 숫자형 변수(일반적 정수형) 
   예) 가족수, 자녀수, 자동차 대수 등 
"""

import matplotlib.pyplot as plt  

# 차트에서 한글 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
# 음수 부호 지원 
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


# 그래프 자료 생성 
data = [127, 90, 201, 150, 250] # 국가별 수출현황 
year = [2010, 2011, 2012, 2013, 2014] # 년도 


# 1. 세로막대 그래프 
plt.bar(x = year, height=data) # 기본색상  
plt.title('국가별 수출현황')
plt.xlabel('년도별')
plt.ylabel('수출현황(단위 : 달러)')
plt.show()


# 2. 가로막대 그래프
plt.barh(y= year, width = data, color='blue') # 색상적용  
plt.title('국가별 수출현황')
plt.xlabel('수출현황(단위 : 달러)')
plt.ylabel('년도별')
plt.show()


# 3. 누적형 세로막대 그래프
cate = ['A', 'B', 'C', 'D'] # 집단 
val1 = [15, 8, 12, 10]  # 첫 번째 데이터셋
val2 = [5, 12, 8, 15]  # 두 번째 데이터셋

plt.bar(cate, val1, label='데이터셋1', alpha=0.5) # 투명도  
plt.bar(cate, val2, bottom=val1, label='데이터셋2', alpha=1)
# bottom : val1 위에 val2 올리기  
plt.title('누적형 막대 그래프')
plt.xlabel('카테고리')
plt.ylabel('값')
plt.legend() # 범례 추가
plt.show()



# 4. 원 그래프 : 비율 적용 
labels = ['싱가폴','태국','한국','일본','미국'] 
print(data) # [127, 90, 201, 150, 250]

plt.pie(x = data, labels = labels) # 100% 
plt.show()


# 비율 계산 
tot = sum(data) # 818
rate = [round((d / tot)*100, 2)  for d in data ] # 백분율   
rate # [15.53, 11.0, 24.57, 18.34, 30.56]

# 새로운 lable 
new_lables = [] 

for i in range(len(labels)) :
    new_lables.append(labels[i] + '\n' + str(rate[i]) + '%')
    

plt.pie(x = data, labels = new_lables) 
plt.show()

###########################################################################################################

"""
 연속형 변수 시각화 : 산점도, 히스토그램, box-plot  
 - 연속형 변수 : 셀수 없는 숫자형 변수(일반적 실수형)
   예) 급여, 나이, 몸무게 등   
"""

import random # 난수 생성 
import statistics as st # 수학/통계 함수 
import matplotlib.pyplot as plt # data 시각화 

# 차트에서 한글 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
# 음수 부호 지원 
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


# 그래프 자료 생성 
data1 = range(-3, 7) # -3 ~ 6
data2 = [random.random() for i in range(10)] # 0~1사이 난수 실수 
data2

# 1. 산점도 그래프 : 2개 변수 이용, 단일 색상  
plt.scatter(x=data1, y=data2, c='r', marker='o')
plt.title('scatter plot')
plt.show()


# 군집별 산점도 : 군집별 색상 적용 
cdata = [random.randint(a=1, b=4) for i in range(10)]  # 난수 정수(1~4) 
cdata # [4, 1, 3, 4, 4, 2, 1, 2, 3, 2]

plt.scatter(x=data1, y=data2, c=cdata, marker='o')
plt.title('scatter plot')
plt.show()


# 군집별 label 추가 
plt.scatter(x=data1, y=data2, c=cdata, marker='o') # 산점도 

for idx, val in enumerate(cdata) : # 색인, 내용
    plt.annotate(text=val, xy=(data1[idx], data2[idx]))
    #plt.annotate(text=텍스트, xy=(x좌표, y좌표)
plt.title('scatter plot') # 제목 
plt.show()



# 2. 히스토그램 그래프 : 1개 변수, 대칭성 확인     
data3 = [random.gauss(mu=0, sigma=1) for i in range(1000)] 
print(data3) # 표준정규분포(-3 ~ +3) 

# 난수 통계
min(data3) # -3.2816997272974784
max(data3) # 3.255899905138625

# 평균과 표준편차 
st.mean(data3) # 0.015763323542712086
st.stdev(data3) # 0.9974539216362369


# 정규분포 시각화 
'''
히스토그램 : x축(계급), y축(빈도수)
'''
help(plt.hist)
plt.hist(data3, label='hist1') # 기본형(계급=10),histtype='bar'  
plt.hist(data3, bins=20, histtype='stepfilled', label='hist2') # 계급, 계단형 적용  
plt.legend(loc = 'best') # 범례
plt.show()
'''
loc 속성
best 
lower left/right
upper left/right
center 
'''


# 3. 박스 플롯(box plot)  : 기초통계 & 이상치(outlier) 시각화
data4 = [random.randint(a=45, b=85) for i in range(100)]  # 45~85 난수 정수 
data4

plt.boxplot(data4)
plt.show()

# 기초통계 : 최솟값/최댓값, 사분위수(1,2,3)
min(data4) # 45
max(data4) # 85

# 사분위수 : q1, q2(중위수), q3
st.quantiles(data4) # [54.0, 65.0, 74.75]

st.median(data4) # 중위수 : 62.0
st.mean(data4) # 평균 :  63.62


# 4. 이상치(outlier) 발견 & 처리 
import pandas as pd 

path = r'C:\ITWILL\4_Python_ML\data'

insurance = pd.read_csv(path + '/insurance.csv')
insurance.info()
'''
 0   age       1338 non-null   int64  : 나이 
 1   sex       1338 non-null   object  : 성별 
 2   bmi       1338 non-null   float64 : 비만도지수 
 3   children  1338 non-null   int64  : 자녀수  
 4   smoker    1338 non-null   object : 흡연유무 
 5   region    1338 non-null   object : 지역
 6   charges   1338 non-null   float64 : 의료비 
'''

# 1) subset 만들기 
df = insurance[['age','bmi']] # 비율척도(절대0점)
df.shape # (1338, 2)

# 2) 이상치 발견과 처리 
df.describe() # # 요약통계량 
'''
               age          bmi
count  1338.000000  1338.000000
mean     39.730194    30.524488
std      20.224425     6.759717
min      18.000000   -37.620000  -> 이상치 
25%      27.000000    26.220000
50%      39.000000    30.332500
75%      51.000000    34.656250
max     552.000000    53.130000 -> 이상치 
'''

# 3) 이상치 시각화 
plt.boxplot(df)
plt.show()


# 4) 이상치 처리 : 100세 이하 -> subset 
new_df = df[df['age'] <= 100] # age 이상치 제거 
new_df.shape # (1335, 2)

plt.boxplot(new_df)
plt.show()


# 5) bmi 이상치 처리 : iqr 방식  
new_df['bmi'].describe()

q1 = 26.22
q3 = 34.6875
iqr = q3 - q1

outlier_step = 1.5 * iqr

minval = q1 - outlier_step # 하한값 
maxval = q3 + outlier_step # 상한값 

# 정상범위 : 13.51 ~ 47.38
minval # 13.518749999999997
maxval #  47.38875

# 정상범위 subset 만들기 
new_df2 = new_df[(new_df.bmi >= minval) & (new_df.bmi <= maxval)] 

new_df2.shape # (1323, 2)

# age, bmi 정상범주 확인 
plt.boxplot(new_df2)
plt.show()

###########################################################################################################

"""
  서브플롯(subplot) 차트 시각화 
"""

import random # 수치 data 생성 
import matplotlib.pyplot as plt # data 시각화 


# 1. subplot 생성 
fig = plt.figure(figsize = (10, 5)) # 차트 크기 지정 
x1 = fig.add_subplot(2,2,1) # 2행2열 1번 
x2 = fig.add_subplot(2,2,2) # 2행2열 2번 
x3 = fig.add_subplot(2,2,3) # 2행2열 3번 
x4 = fig.add_subplot(2,2,4) # 2행2열 4번 

# 2.차트 데이터 생성 
data1 = [random.gauss(mu=0, sigma=1) for i in range(100)] # 정규분포 난수 100개  
data2 = [random.randint(1, 100) for i in range(100)] # 1 ~ 100 난수 정수 100개 
cdata = [random.randint(1, 4) for i in range(100)] # 1 ~ 4


# 3. 각 격차에 차트 크리기 
x1.hist(data1) # 히스토그램 
x2.scatter(data1, data2, c=cdata) # 산점도 
x3.plot(data2) # 기본 차트 
x4.plot(data1, data2, 'g--') # 기본 차트 : 선 색과 스타일 

# subplot 수준 제목 적용 
x1.set_title('hist', fontsize=15)
x2.set_title('scatter', fontsize=15)
x3.set_title('default plot', fontsize=15)
x4.set_title('color plot', fontsize=15)

# figure 수준 제목 적용 
fig.suptitle('suptitle title', fontsize=20)
plt.show()

###########################################################################################################

"""
 - marker, color, line style, label 이용 
"""

import random # 수치 data 생성 
import matplotlib.pyplot as plt # data 시각화 
plt.style.use('ggplot') # 차트내 격차 제공 


# 1. data 생성 : 정규분포
data1 = [random.gauss(mu=0.5, sigma=0.3) for i in range(100)] 
data2 = [random.gauss(mu=0.7, sigma=0.2) for i in range(100)] 
data3 = [random.gauss(mu=0.1, sigma=0.9) for i in range(100)]   

# 2. Fugure 객체 
fig = plt.figure(figsize = (12, 5)) 
chart = fig.add_subplot()  # 1개 격자

# 3. plot : 시계열 시각화 
chart.plot(data1, marker='o', color='blue', linestyle='-', label='data1')
chart.plot(data2, marker='+', color='red', linestyle='--', label='data2')
chart.plot(data3, marker='*', color='green', linestyle='-.', label='data3')
plt.title('Line plots : marker, color, linestyle')
plt.xlabel('index')
plt.ylabel('random number')
plt.legend(loc='best')
#plt.show()


# 4. image save & read  
plt.savefig(r"ts_plot.png") # 이미지 파일 저장 
plt.show()


# 이미지 파일 읽기 
import matplotlib.image as img 

image = img.imread(r"ts_plot.png") # 이미지 파일 읽기 
plt.imshow(image)

image.shape # (360, 864, 4) : (h, w, color)


###########################################################################################################

"""
Pandas 객체 시각화 : 이산형 변수 시각화  
  형식) object.plot(kind='유형', 속성)
      object : Series, DataFrame
      kind : bar, pie, scatter, hist, box
      속성 : 제목, 축이름 등 
"""

import pandas as pd # Series, DataFrame 
import numpy as np  
import matplotlib.pyplot as plt 

# 1. 기본 차트 시각화 

# 1) Series 객체 시각화 : 1d  
ser = pd.Series(np.random.randn(10),
          index = np.arange(0, 100, 10))
ser
dir(ser) # plot

ser.plot() # 선 그래프 
plt.show()

# 2) DataFrame 객체 시각화 : 2d
df = pd.DataFrame(np.random.randn(10, 4),
                  columns=['one','two','three','fore'])
df

# 기본 차트 : 선 그래프 
df.plot()  
plt.show()

# 막대차트 
help(df.plot)

df.plot(kind = 'bar', title='bar chart')
plt.show()


# 2. dataset 이용 
path = r'C:\ITWILL\4_Python_ML\data'

tips = pd.read_csv(path + '/tips.csv')
tips.info()
'''
 0   total_bill  244 non-null    float64
 1   tip         244 non-null    float64
 2   sex         244 non-null    object 
 3   smoker      244 non-null    object 
 4   day         244 non-null    object : 행사 요일 
 5   time        244 non-null    object 
 6   size        244 non-null    int64  : 행사 규모 
'''
 

# 행사 요일별 : 파이 차트 
cnt = tips['day'].value_counts() # 4개 
type(cnt) # pandas.core.series.Series

cnt.plot(kind = 'pie')
plt.show()

tips['size'].value_counts() # 6개 
'''
2    156
3     38
4     37
5      5
1      4
6      4
'''


# 요일(day) vs 규모(size) : 교차분할표(카이제곱검정 도구) 
table = pd.crosstab(index=tips['day'], 
                    columns=tips['size'])

table
'''
size  1   2   3   4  5  6
day                      
Fri   1  16   1   1  0  0
Sat   2  53  18  13  1  0
Sun   0  39  15  18  3  1
Thur  1  48   4   5  1  3
'''
type(table) # pandas.core.frame.DataFrame

# 개별 막대차트 
table.plot(kind='bar')

# size : 2~5칼럼으로 subset 
new_table = table.loc[:,2:5]
new_table

# 누적형 가로막대차트 
new_table.plot(kind='barh', stacked=True,
               title = 'day vs size')


###########################################################################################################

"""
Pandas 객체 시각화 : 연속형 변수 시각화  
 - hist, kde, scatter, box 등 
"""

import pandas as pd
import numpy as np # dataset 
import matplotlib.pyplot as plt # chart

# file 경로 
path = r'C:\ITWILL\4_Python_ML\data'

# 1. 산점도 
dataset = pd.read_csv(path + '/dataset.csv')
print(dataset.info())

# 연속형 변수 : x=age, y=price 
plt.scatter(dataset['age'], dataset['price'], 
            c=dataset['gender'])
plt.show()


# 2. hist, kde, box

# DataFrame 객체 
df = pd.DataFrame(np.random.randn(100, 4),
               columns=('one','two','three','fore'))

# 1) 히스토그램
df['one'].plot(kind = 'hist', title = 'histogram')
plt.show()

# 2) 커널밀도추정 
df['one'].plot(kind = 'kde', title='kernel density plot')
plt.show()

# 3) 박스플롯
df.plot(kind='box', title='boxplot chart')
plt.show()


# 3. 산점도 행렬 
from pandas.plotting import scatter_matrix

# 3) iris.csv
iris = pd.read_csv(path + '/iris.csv')
cols = list(iris.columns)

x = iris[cols[:4]] 
x

# 산점도 matrix 
scatter_matrix(x)
plt.show()


###########################################################################################################

"""
 - 산점도 행렬과 3차원 산점도 
"""

import pandas as pd # object
import matplotlib.pyplot as plt # chart

path = r'C:\ITWILL\4_Python_ML\data'

# 1. 산점도 행렬 
from pandas.plotting import scatter_matrix

# 3) iris.csv
iris = pd.read_csv(path + '/iris.csv')
cols = list(iris.columns)

x = iris[cols[:4]]
print(x.head())

# 산점도 matrix 
scatter_matrix(x)
plt.show()


# 2. 3차원 산점도 
from mpl_toolkits.mplot3d import Axes3D

col_x = iris[cols[0]] # 1번 
col_y = iris[cols[1]] # 2번 
col_z = iris[cols[2]] # 3번 

cdata = [] # color data 
for s in iris[cols[-1]] : # 'Species'
    if s == 'setosa' :
        cdata.append(1)
    elif s == 'versicolor' :
        cdata.append(2)
    else :
        cdata.append(3)

fig = plt.figure()
chart = fig.add_subplot(projection='3d') # Axes3D

chart.scatter(col_x, col_y, col_z, c = cdata)
chart.set_xlabel('Sepal.Length')
chart.set_ylabel('Sepal.Width')
chart.set_zlabel('Petal.Length')
plt.show()

###########################################################################################################

'''
시계열 데이터 시각화 
'''

import pandas as pd
import matplotlib.pyplot as plt


# 1. 날짜형식 수정(다국어)
path = r'C:/ITWILL/4_Python_ML/data'
cospi = pd.read_csv(path + "/cospi.csv")
cospi.info()
'''
 0   Date    247 non-null    object
 1   Open    247 non-null    int64 
 2   High    247 non-null    int64 
 3   Low     247 non-null    int64 
 4   Close   247 non-null    int64 
 5   Volume  247 non-null    int64 
'''
cospi.head() 

# object -> Date형 변환 
cospi['Date'] = pd.to_datetime(cospi['Date'])
print(cospi.info())
# 0   Date    247 non-null    datetime64[ns]

cospi.head() # 2016 
cospi.tail() # 2015

# Date 칼럼 : subset 만들기(2016-01-01 ~ 2016-02-26)
new_cospi = cospi[(cospi.Date >= '2016-01') & (cospi.Date <= '2016-02-26')]
new_cospi


# 2. 시계열 데이터/시각화

# 1개 칼럼 추세그래프 
cospi['High'].plot(title = "Trend line of High column")
plt.show()

# 2개 칼럼(중첩list) 추세그래프
cospi[['High', 'Low']].plot(color = ['r', 'b'],
        title = "Trend line of High and Low column")
plt.show() 


# index 수정 : Date 칼럼 이용  
new_cospi = cospi.set_index('Date')
print(new_cospi.info())
print(new_cospi.head())

# 날짜형 색인 
new_cospi.index #  DatetimeIndex(['2016-02-26', '2016-02-25',

dir(new_cospi)
'''
 'sort_index', : 색인 정렬 
 'sort_values' : 특정 칼럼으로 정렬 
''' 

# 색인으로 오름차순 정렬 
new_cospi = new_cospi.sort_index()

print(new_cospi.loc['2016']) # 년도 선택 
print(new_cospi.loc['2016-02']) # 월 선택 
print(new_cospi.loc['2016-01':'2016-02']) # 범위 선택 

# 2016년도 주가 추세선 시각화 
new_cospi_HL = new_cospi[['High', 'Low']]
new_cospi_HL.loc['2016'].plot(title = "Trend line of 2016 year")
plt.show()

new_cospi_HL.loc['2016-02'].plot(title = "Trend line of 2016 year")
plt.show()


# 3. 이동평균(평활) : 지정한 날짜 단위 평균계산 -> 추세그래프 스무딩  

# 5일 단위 평균계산 : 평균계산 후 5일 시작점 이동 
roll_mean5 = pd.Series.rolling(new_cospi.High,
                               window=5, center=False).mean()
print(roll_mean5)

# 10일 단위 평균계산 : 평균계산 후 10일 시작점 이동
roll_mean10 = pd.Series.rolling(new_cospi.High,
                               window=10, center=False).mean()


# 1) High 칼럼 시각화 
new_cospi['High'].plot(color = 'blue', label = 'High column')


# 2) rolling mean 시각화 : subplot 이용 - 격자 1개  
fig = plt.figure(figsize=(12,4))
chart = fig.add_subplot()
chart.plot(new_cospi['High'], color = 'blue', label = 'High column')
chart.plot(roll_mean5, color='red',label='5 day rolling mean')
chart.plot(roll_mean10, color='green',label='10 day rolling mean')

plt.legend(loc='best')
plt.show()

###########################################################################################################

"""
Seaborn : Matplotlib 기반 다양한 배경 테마, 통계용 차트 제공 
커널 밀도(kernel density), 카운트 플롯, 다차원 실수형 데이터,2차원 카테고리 데이터
2차원 복합 데이터(box-plot), heatmap, catplot 
"""

import seaborn as sn # 별칭 

# 1. 데이터셋 확인 
names = sn.get_dataset_names() 
print(names)
len(names) # 88

# 2. 데이터셋 로드 
iris = sn.load_dataset('iris')
type(iris) # pandas.core.frame.DataFrame
print(iris.info())
'''
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object 
'''

tips = sn.load_dataset('tips')
tips.info()

###########################################################################################################

"""
1. Object vs Category 
  - object : 문자열 순서 변경 불가 
  - category : 문자열 순서 변경 가능 
2. 범주형 자료 시각화 
"""

import matplotlib.pyplot as plt
import seaborn as sn


# 1. Object vs Category 

# dataset load
titanic = sn.load_dataset('titanic')

print(titanic.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
'''
    
# subset 만들기 
df = titanic[['survived','age','class','who']]
df.info()
'''
 0   survived  891 non-null    int64   
 1   age       714 non-null    float64 
 2   class     891 non-null    category
 3   who       891 non-null    object 
'''
df.head()
'''
   survived   age  class    who
0         0  22.0  Third    man
1         1  38.0  First  woman
2         1  26.0  Third  woman
3         1  35.0  First  woman
4         0  35.0  Third    man
'''

# category형 정렬
dir(df)
'''
 sort_index() : 행이름 정렬 
 sort_values(by='칼럼명') : 칼럼값 정렬 
''' 

df.sort_values(by = 'class') # category 오름차순
# First > Second > Third

# object형 정렬 
df.sort_values(by = 'who') # object 오름차순 
# child > man > woman


# category형 변수 순서 변경 : Third > Second > First 
df['class_new'] = df['class'].cat.set_categories(['Third', 'Second', 'First'])
df.info()

df.sort_values(by = 'class_new')

# object -> category형 변환 
df['who'].dtype # 자료형 확인 : dtype('O') : object
df['who_new'] = df['who'].astype('category') # 자료형 변환 
df.info()


# 2. 범주형 자료 시각화 

# 1) 배경 스타일 
sn.set_style(style='darkgrid')

tips = sn.load_dataset('tips')
print(tips.info())

tips.smoker.value_counts()
'''
No     151
Yes     93
'''

# 2) category형 자료 시각화 
sn.countplot(x = 'smoker', data = tips) # 빈도수 + 막대차트 
plt.title('smoker of tips')
plt.show()

# 행사일 
tips.day.value_counts()
'''
Sat     87
Sun     76
Thur    62
Fri     19
'''
sn.countplot(x = 'day', data = tips) # 빈도수 + 막대차트 
plt.title('day of tips')
plt.show()

###########################################################################################################

'''
 연속형 변수 시각화 
 - 산점도, 산점도 행렬, boxplot 
'''

import matplotlib.pyplot as plt
import seaborn as sn

# seaborn 한글과 음수부호, 스타일 지원 
sn.set(font="Malgun Gothic", 
            rc={"axes.unicode_minus":False}, style="darkgrid")

# dataset load 
iris = sn.load_dataset('iris')
tips = sn.load_dataset('tips')


x = iris.sepal_length
x

# 1-1. displot : 히스토그램
sn.displot(data=iris, x='sepal_length', kind='hist')  
plt.title('iris Sepal length hist') # 단위 : Count 
plt.show()


# 1-2. displot : 밀도분포곡선 : hue='범주형'
sn.displot(data=iris, x='sepal_length', kind="kde", hue='species') 
plt.title('iris Sepal length kde') # 단위 : Density
plt.show()


# 2. 산점도 행렬(scatter matrix)  
sn.pairplot(data=iris, hue='species') 
plt.show()


# 3. 산점도 : 연속형+연속형   
sn.scatterplot(x="sepal_length", y="petal_length", data=iris)
plt.title('산점도(scatter)')
plt.show()

# 2차 : 연속형+연속형+범주형(hue=집단변수) 
sn.scatterplot(x="sepal_length", y="petal_length", 
               hue="species", data=iris)
plt.title('산점도(scatter)')
plt.show()


# 4. boxplot : 범주형+연속형=범주형
sn.boxplot(x="day", y="total_bill", hue="sex", data=tips)
plt.title("성별을 기준으로 요일별 총지불 금액")
plt.show()

###########################################################################################################

"""
 - 분석모델 관련 시각화
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn

# seaborn 한글과 음수부호, 스타일 지원 
sn.set(font="Malgun Gothic", 
            rc={"axes.unicode_minus":False}, style="darkgrid")


# dataset 로드 
flights = sn.load_dataset('flights')
flights.info()
'''
 0   year        144 non-null    int64    -> 년도 : x  
 1   month       144 non-null    category -> 월 : 집단변수 
 2   passengers  144 non-null    int64    -> 탑승객 : y
'''
144 / 12 # 12년도 

iris = sn.load_dataset('iris')
iris.info()

# 1. 오차대역폭을 갖는 시계열 : x:시간축, y:통계량 
sn.lineplot(x = 'year', y = 'passengers', data = flights)
plt.show()

# hue 추가 
sn.lineplot(x = 'year', y = 'passengers', hue='month',
            data = flights)
plt.show()
 

# 2. 선형회귀모델 : 산점도 + 회귀선 
sn.regplot(x = 'sepal_length', y = 'petal_length', 
           data = iris)  
plt.show()


# 3. heatmap : 분류분석 평가 
y_true = pd.Series([1,0,1,1,0]) # 정답 
y_pred = pd.Series([1,0,0,1,0]) # 예측치 

# 1) 교차분할표(혼동 행렬) 
tab = pd.crosstab(y_true, y_pred, 
            rownames=['관측치'], colnames=['예측치'])
tab

# 2) heatmap
sn.heatmap(data=tab, annot = True) # annot = True : box에 빈도수 
plt.show()





































































 























































