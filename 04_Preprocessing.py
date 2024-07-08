######################################
### 결측치 처리
######################################

'''
 - 결측치 확인 및 처리(제거 및 채우기) 
'''

import pandas as pd
path = r'C:\ITWILL\4_Python_ML\data'
data = pd.read_csv(path +'/dataset.csv') 
data.info()
'''
 0   resident  217 non-null    int64  
 1   gender    217 non-null    int64  
 2   job       205 non-null    float64
 3   age       217 non-null    int64  
 4   position  208 non-null    float64
 5   price     217 non-null    float64
 6   survey    217 non-null    int64  
'''
 

# 1. 칼럼단위 결측치(NaN) 확인  
data.isnull().any() # 결측치 유무  
data.isnull().sum() # 결측치 총개수 
'''
job         12
position     9
'''
data.shape # (217, 7)  


# 2. 전체 칼럼 기준 결측치 제거 
new_data = data.dropna() # 모든 결측치 행 제거 
new_data.shape # (198, 7)
217 - 198 # 19

# 결측치 제거 확인 
new_data.isnull().sum()


# 3. 특정 칼럼 기준 결측치 제거   
# 형식) DF.dropna(subset = ['칼럼명1', '칼럼명2'])
new_data2 = data.dropna(subset = ['job']) # job 기준  
new_data2.shape # (205, 7) 
217 - 205 # 121

new_data2.isnull().sum()
# position    7 : 다른 칼럼 삭제 안됨 


# 4. 모든 결측치 다른값으로 채우기 : 상수 or 통계
# 형식) DF.fillna(값)  
new_data3 = data.fillna(0.0) 
new_data3.shape  # (217, 7)

new_data3.isna().sum()


# 5-1. 특정변수 결측치 채우기 : 숫자변수(상수 or 통계 대체) 
new_data4 = data.copy() # 내용복제
new_data4.isna().sum() 
'''
job         12
position     9
'''

# position 결측치 평균 대체
new_data4['position'].fillna(new_data4['position'].mean(), inplace=True)
# inplace=True : 현재 객체 반영
 
new_data4.isna().sum()   
# job         12
# position     0


# 5-2. 특정변수 결측치 채우기 : 범주형변수(빈도수가 높은 값으로 대체)  
new_data5 = data.copy() # 내용복제 
new_data5['job'].unique() # [ 1.,  2., nan,  3.]
new_data5['job'].value_counts()
'''
3.0    77 -> 빈도수가 높은 값 대체 
2.0    74
1.0    54
'''

new_data5['job'].fillna(3.0, inplace=True) # 현재 객체 반영 
new_data5.isnull().sum()


# 6. 결측치 비율 40% 이상인 경우 해당 컬럼 제거 
data.isna().sum() # job 칼럼 제거 

# job 칼럼 결측치 비율 
12 / len(data) # 5.5%

# job 칼럼 제거 : DF.drop(['칼럼명1', '칼럼명2'], axis = 1)
new_data6 = data.drop(['job'], axis = 1) # 열축 방향  
new_data6.shape # (217, 6)

new_data6.head()

##################################################################################################################

######################################
### 결측치 처리
######################################

'''
- 특수문자를 결측치로 처리하는 방법 
'''

import pandas as pd 
pd.set_option('display.max_columns', 50) # 최대 50 칼럼수 지정

# 데이터셋 출처 : https://www.kaggle.com/uciml/breast-cancer-wisconsin-data?select=data.csv
cencer = pd.read_csv(r'C:\ITWILL\4_Python_ML\data\brastCencer.csv')
cencer.info()
'''
RangeIndex: 699 entries, 0 to 698
Data columns (total 11 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   id               699 non-null    int64  -> 제거(구분자) 
 1   clump            699 non-null    int64 
 2   cell_size        699 non-null    int64 
 3   cell_shape       699 non-null    int64 
 4   adhesion         699 non-null    int64 
 5   epithlial        699 non-null    int64 
 6   bare_nuclei      699 non-null    object -> x변수(노출원자핵) : 숫자형 변환
 7   chromatin        699 non-null    int64 
 8   normal_nucleoli  699 non-null    int64 
 9   mitoses          699 non-null    int64 
 10  class            699 non-null    int64 -> y변수 
'''


# 1. 변수 제거 
df = cencer.drop(['id'], axis = 1) # 열축 기준 : id 칼럼 제거  
df.shape # (699, 10)


# 2. x변수 숫자형 변환 : object -> int형 변환
df['bare_nuclei'] 

# DF['칼럼명'].astype('자료형') : Object -> int
df['bare_nuclei'] = df['bare_nuclei'].astype('int') # error 발생 
# ValueError: invalid literal for int() with base 10: '?'


# 3. 특수문자 결측치 처리 & 자료형 변환 

# 1) 특수문자 결측치 대체 : '?' <- NaN  
import numpy as np 
df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan) # ('old','new')

# 2) 전체 칼럼 단위 결측치 확인 
df.isnull().any() 
df.isnull().sum() # 16
# bare_nuclei         True

# 3) 결측치 제거  
new_df = df.dropna(subset=['bare_nuclei'])    
new_df.shape # (683, 10) : 16개 제거 


# 4) int형 변환 
new_df['bare_nuclei'] = new_df['bare_nuclei'].astype('int64') 
new_df.info()


new_df['class'].unique() # [2, 4] -> [0, 1]


# 4. y변수 레이블 인코딩 : 10진수 변환 
from sklearn.preprocessing import LabelEncoder # class  

# 인코딩 객체 
encoder = LabelEncoder().fit(new_df['class']) # 1) 객체에 반영  

# data변환 : [2, 4] -> [0, 1]
labels = encoder.transform(new_df['class']) # 2) 자료 변형  
labels # 0 or 1

# 칼럼 추가 
new_df['y'] = labels

# class 변수 제거 
new_df = new_df.drop(['class'], axis = 1)

new_df.info()
'''
 0   clump            683 non-null    int64
 1   cell_size        683 non-null    int64
 2   cell_shape       683 non-null    int64
 3   adhesion         683 non-null    int64
 4   epithlial        683 non-null    int64
 5   bare_nuclei      683 non-null    int64
 6   chromatin        683 non-null    int64
 7   normal_nucleoli  683 non-null    int64
 8   mitoses          683 non-null    int64
 9   y                683 non-null    int64
'''

##################################################################################################################


######################################
### 2. 이상치 처리 
######################################
"""
 이상치(outlier) 처리 : 정상범주에서 벗어난 값(극단적으로 크거나 작은 값) 처리  
  - IQR(Inter Quentile Range) 방식으로 탐색과 처리   
"""

import pandas as pd 

data = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\insurance.csv")
data.info()
'''
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   region    1338 non-null   object 
 6   charges   1338 non-null   float64
'''
 
# 1. 범주형 이상치 탐색  
data.sex.unique() # ['female', 'male']
data.smoker.unique() # ['yes', 'no']
data.region.unique() # ['southwest', 'southeast', 'northwest', 'northeast']


# 2. 숫자형 변수 이상치 탐색  
des = data.describe() # 요약통계량 
print(des)
'''
               age          bmi     children       charges
count  1338.000000  1338.000000  1338.000000   1338.000000
mean     39.730194    30.524488     1.094918  13270.422265
std      20.224425     6.759717     1.205493  12110.011237
min      18.000000   -37.620000     0.000000   1121.873900
25%      27.000000    26.220000     0.000000   4740.287150
50%      39.000000    30.332500     1.000000   9382.033000
75%      51.000000    34.656250     2.000000  16639.912515
max     552.000000    53.130000     5.000000  63770.428010

age : 최대값 이상치 
bmi : 최소값 이상치
charges : 불투명(IQR 적용) 
'''

# 3. boxplot 이상치 탐색 
import matplotlib.pyplot as plt

plt.boxplot(data['age']) # age 최대값 이상치  
plt.show()

plt.boxplot(data['bmi']) # bmi 최소값/최대값 이상치  
plt.show()

plt.boxplot(data['charges']) # charges 최대값 이상치 
plt.show()


# 4. 이상치 처리 : 제거 & 대체 

# 1) bmi 이상치 제거 
df = data.copy() # 복제
data.shape # (1338, 7)

df = df[df['bmi'] > 0] # 음수 이상치 처리 
df.shape # (1335, 7)


# 2) age 이상치 대체   
df = data.copy() # 복제

df[df['age'] > 100] # 100세 이상   

# 100세 이상 -> 100세 대체 
df.loc[df.age > 100, 'age'] = 100 # 현재 객체 반영 
df.shape # (1338, 7)


# 5. IQR방식 이상치 발견 및 처리 :  변수의 의미를 모르는 경우 

# 1) IQR방식으로 이상치 발견   
'''
 IQR = Q3 - Q1 : 제3사분위수 - 제1사분위수
 outlier_step = 1.5 * IQR
 정상범위 : Q1 - outlier_step ~ Q3 + outlier_step
'''  

Q3 = des.loc['75%', 'age'] 
Q1 = des.loc['25%', 'age'] 
IQR = Q3 - Q1

outlier_step = 1.5 * IQR # 36.0

minval = Q1 - outlier_step # 하한값 
maxval = Q3 + outlier_step # 상한값 
print(f'minval : {minval}, maxval : {maxval}') 
# minval : -9.0, maxval : 87.0

# 비율척도 : 하한값 = 0
minval = 0


# 2) 이상치 제거  
df = data.copy() # 복제 

df = df[(df['age'] >= minval) & (df['age'] <= maxval)]
df.shape # (1335, 7)


# 나이 시각화 
df['age'].plot(kind='box')


# 3) 이상치 대체 
df = data.copy() # 복제

# 하한값으로 대체 
df.loc[df['age'] < minval, 'age'] = minval  

# 상한값으로 대체 
df.loc[df['age'] > maxval, 'age' ] = maxval

df.shape # (1338, 7)

# 나이 시각화 
df['age'].plot(kind='box')

'''
평균과 표준편차 이용 : ppt.14
    하한값=평균 - n*표준편차,
    상한값=평균 + n*표준편차
    (n=3 : threshold)
'''
# age 칼럼 기준 
df = data.copy() # 복제

avg = df['age'].mean()
std = df['age'].std()
n = 3 

minval = 0 # avg - n*std
maxval = avg + n*std # 100.40346993181589

df = df[(df['age'] >= minval) & (df['age'] <= maxval)]
df.shape # (1335, 7)

##################################################################################################################

######################################
### 3. 데이터 인코딩 
######################################

"""
데이터 인코딩 : 머신러닝 모델에서 범주형변수를 대상으로 숫자형의 목록으로 변환해주는 전처리 작업
 - 방법 : 레이블 인코딩(label encoding), 원-핫 인코딩(one-hot encoding)   
 - 레이블 인코딩(label encoding) : 트리계열모형(의사결정트리, 앙상블)의 변수 대상(10진수) 
 - 원-핫 인코딩(one-hot encoding) : 회귀계열모형(선형,로지스틱,SVM,신경망)의 변수 대상(2진수) 
   -> 회귀모형에서는 인코딩값이 가중치로 적용되므로 원-핫 인코딩으로 변환  
"""


import pandas as pd 

data = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\skin.csv")
data.info()
'''
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   cust_no      30 non-null     int64  -> 변수 제외(구분자 : id, no, name) 
 1   gender       30 non-null     object -> x변수 
 2   age          30 non-null     int64 
 3   job          30 non-null     object
 4   marry        30 non-null     object
 5   car          30 non-null     object
 6   cupon_react  30 non-null     object -> y변수(쿠폰 반응) 
''' 


## 1. 변수 제거 : cust_no
df = data.drop('cust_no', axis = 1)
df.info()


### 2. 레이블 인코딩 : 트리모델 계열의 x, y변수 인코딩  
from sklearn.preprocessing import LabelEncoder # 인코딩 도구 

# 1) 쿠폰 반응 범주 확인 
df.cupon_react.unique() # array(['NO', 'YES'], dtype=object) 


# 2) 인코딩
encoder = LabelEncoder() # encoder 객체 
label = encoder.fit_transform(df['cupon_react']) # data 반영 
label # 0 or 1 

# 3) 칼럼 추가 
df['label'] = label
df = df.drop('cupon_react', axis = 1) # 기존 칼럼 제거 
df.info()


### 3. 원-핫 인코딩 : 회귀모델 계열의 x변수 인코딩  

# 1) k개 목록으로 가변수(더미변수) 만들기 
df_dummy = pd.get_dummies(data=df, dtype='uint8') # 기준변수 포함 
df_dummy # age  gender_female  gender_male  ...  marry_YES  car_NO  car_YES -> (1+8)
df_dummy.info()
# 가변수 : 칼럼명_범주명 
df_dummy.shape # (30, 10)


# 2) 특정 변수 선택   
df_dummy2 = pd.get_dummies(data=df, columns=['label','gender','job','marry'],
                           dtype='uint8')
df_dummy2.info() # Data columns (total 12 columns):

    
# 3) k-1개 목록으로 가변수(더미변수) 만들기   
df_dummy3 = pd.get_dummies(data=df, drop_first=True, dtype='uint8') # 기준변수 제외(권장)
df_dummy3.shape #  (30, 6)  
# 기준변수 : 영문자,한글 오름차순 
df_dummy3.info()

###################################
## gender변수 : 순서 변경 -> 가변수 
###################################
df.info()

df['gender'] = df['gender'].astype('category')
df['gender'].unique() # ['male', 'female']

# 순서 변경 
df['gender'] = df['gender'].cat.set_categories(['male','female'])

df_dummy4 = pd.get_dummies(data=df, columns=['gender'], drop_first=True)
df_dummy4.info()

###############################
## 가변수 기준(base) 변경하기  
###############################
import seaborn as sn 
iris = sn.load_dataset('iris')
iris.info()
iris['species'].unique() # ['setosa', 'versicolor', 'virginica']


# 1. 가변수(dummy) : k-1개 
iris_dummy = pd.get_dummies(data = iris, columns=['species'], 
                            drop_first=True)
# drop_first=True : 첫번째 범주 제외(기준변수)
iris_dummy.info()

# 2. base 기준 변경 : 범주 순서변경('virginica' -> 'versicolor' -> 'setosa') 
# object -> category 변환 
iris['species'] = iris['species'].astype('category')

iris['species'] = iris['species'].cat.set_categories(['virginica','versicolor','setosa'])


# 3. 가변수(dummy) : k-1개 
iris_dummy2 = pd.get_dummies(data=iris, columns=['species'], 
                             drop_first=True)
iris_dummy2.info()


##################################################################################################################

#########################################
### 4. 피처 스케일링(feature scaling) 
#########################################

"""
피처 스케일링 : 서로 다른 크기(단위)를 갖는 X변수(feature)를 대상으로 일정한 범위로 조정하는 전처리 작업 
 - 방법 : 표준화, 최소-최대 정규화, 로그변환    
 
 1. 표준화 : X변수를 대상으로 정규분포가 될 수 있도록 평균=0, 표준편차=1로 통일 시킴 
   -> 회귀모델, SVM 계열은 X변수가 정규분포라고 가정하에 학습이 진행되므로 표준화를 적용   
 2. 최소-최대 정규화 : 서로 다른 척도(값의 범위)를 갖는 X변수를 대상으로 최솟값=0, 최댓값=1로 통일 시킴 
   -> 트리모델 계열(회귀모델 계열이 아닌 경우)에서 서로 다른 척도를 갖는 경우 적용 
 3. 로그변환 : log()함수 이용하여 로그변환   
   -> 비선형(곡선) -> 선형(직선)으로 변환
   -> 왜곡을 갖는 분포 -> 좌우대칭의 정규분포로 변환   
"""

# 1. 함수형 스케일링 도구  
from sklearn.preprocessing import scale # 표준화 
from sklearn.preprocessing import minmax_scale # 정규화
import numpy as np # 로그변환 + 난수

# 실습 data 생성 : 난수 정수 생성  
np.random.seed(12) # 시드값 
X = np.random.randint(-10, 100, (5, 4)) # -10~100
X.shape # (5, 4)

X.mean(axis = 0)
# [27.4, 36.4, 44.4, 31.2]


# 1) 표준화 : 평균=0, 표준편차=1 -> -3 ~ +3 조정 
X_zscore = scale(X)
X_zscore.mean(axis = 0)
# [ 8.8817842e-17,  8.8817842e-17,  0.0000000e+00, -4.4408921e-17]
X_zscore.std(axis = 0)
# [1., 1., 1., 1.]

# 2) 정규화 
X_nor = minmax_scale(X) # 0 ~ 1 조정 
X_nor


# 3) 로그변환 
X_log = np.log(X) 
X_log # 음수값 -> nan 

np.log(0) # -inf
np.log(0.5) # -0.6931471805599453
np.log(1) # 0.0
np.log(-4) # nan

# inf, nan 문제 해결 
X_log = np.log1p(np.abs(X)) # np.log(|0+1|)
print(X_log)


# 2. 클래스형 스케일링 도구 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

import pandas as pd
iris = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\iris.csv")
iris.info()


# 1) DataFrame 표준화 
iris.iloc[:,:4].mean(axis = 0)
'''
Sepal.Length    5.843333
Sepal.Width     3.057333
Petal.Length    3.758000
Petal.Width     1.199333
'''

scaler = StandardScaler() # object 생성 
X_scaled = scaler.fit_transform(X=iris.drop('Species', axis=1))# Species 칼럼 제외 
X_scaled.mean(axis = 0)
# [-4.73695157e-16, -7.81597009e-16, -4.26325641e-16, -4.73695157e-16]


# 2) DataFrame 정규화 
scaler2 = MinMaxScaler() # object 생성
X_scaled2 = scaler2.fit_transform(X = iris.drop('Species', axis = 1))

X_scaled2.min(axis = 0) # [0., 0., 0., 0.]
X_scaled2.max(axis = 0) # [1., 1., 1., 1.]


# y변수 인코딩 
y = iris.Species

from sklearn.preprocessing import LabelEncoder

y_encode = LabelEncoder().fit_transform(y)
y_encode # 0 ~ 2


# numpy -> pandas 
new_df = pd.DataFrame(X_scaled, columns=iris.columns[:4])

# y칼럼 추가 
new_df['Species'] = y_encode

new_df.info()

##################################################################################################################

"""
탐색적자료분석(Exploratory Data Analysis)
 - 수집 데이터를 다양한 각도에서 관찰하고 이해하는 과정
 - 그래프나 통계적 방법으로 자료를 직관적으로 파악하는 과정
 - 파생변수 생성, 독립변수와 종속변수 관계 탐색  

 예) 포루투갈의 2차교육과정에서 학생들의 음주에 영향을 미치는 요소는 무엇인가? 
"""

import pandas as pd

## data source : https://www.kaggle.com/uciml/student-alcohol-consumption  
path = r'C:\ITWILL\4_Python_ML\data'
student = pd.read_csv(path + '/student-mat.csv')
student.info() # total 33 columns

'''
학생들의 음주에 미치는 영향을 조사하기 위해 6가지의 변수 후보 선정
독립변수 : sex(성별), age(15~22), Pstatus(부모거주여부), failures(수업낙제횟수), famrel(가족관계), grade(G1+G2+G3 : 연간성적) 
          grade : 0~60(60점이 고득점), Alcohol : 0~500(100:매우낮음, 500:매우높음)으로 가공
종속변수 : Alcohol = (Dalc+Walc)/2*100 : 1주간 알코올 섭취정도  
'''

# 1. subset 만들기 
df = student[['sex','age','Pstatus','failures','famrel','Dalc','Walc','G1','G2','G3']]
df.info()
'''
RangeIndex: 395 entries, 0 to 394
Data columns (total 10 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   sex       395 non-null    object : 성별(F, M)
 1   age       395 non-null    int64  : 나이(15 ~ 22)
 2   Pstatus   395 non-null    object : 부모거주여부(T, A)
 3   failures  395 non-null    int64  : 수업낙제횟수(0,1,2,3)
 4   famrel    395 non-null    int64  : 가족관계(1,2,3,4,5)
 5   Dalc      395 non-null    int64  : 1일 알콜 소비량(1,2,3,4,5)   
 6   Walc      395 non-null    int64  : 1주일 알콜 소비량(1,2,3,4,5)  
 7   G1        395 non-null    int64  : 첫번째 학년(0~20)
 8   G2        395 non-null    int64  : 두번째 학년(0~20) 
 9   G3        395 non-null    int64  : 마지막 학년(0~20) 
'''

# 1. 문자형 변수의 빈도수와 숫자형 변수 통계량 확인  
df.sex.value_counts() # 문자형 변수 
df.Pstatus.value_counts() # 문자형 변수 

df.describe() # 숫자형 변수 


# 2. 파생변수 만들기 
grade = df.G1 + df.G2 + df.G3 # 성적 
grade.describe() # 4 ~ 58 

Alcohol = (df.Dalc + df.Walc) / 2 * 100 # 알콜 소비량 
Alcohol.describe() # 100 ~ 500(100:매우낮음, 500:매우높음)

# 1) 파생변수 추가 
df['grade'] = grade 
df['Alcohol'] = Alcohol


# 2) 기존 변수 제거
new_df = df.drop(['Dalc','Walc','G1','G2','G3'], axis = 1) # 칼럼 기준 제거 
new_df.info()
'''
 0   sex       395 non-null    object : x변수 
 1   age       395 non-null    int64  
 2   Pstatus   395 non-null    object 
 3   failures  395 non-null    int64  
 4   famrel    395 non-null    int64  
 5   grade     395 non-null    int64  
 6   Alcohol   395 non-null    float64 : y변수 
''' 

new_df.head()

import seaborn as sn # 시각화 도구 
import matplotlib.pyplot as plt # plt.show()


# 3. EDA : 종속변수(Alcohol) vs 독립변수 탐색 

### 연속형(y) vs 명목형(x)  

new_df.sex.value_counts()
'''
F    208
M    187
'''
# 1) Alcohol vs sex
sn.countplot(x='sex',  data=new_df) # 명목형 변수 빈도수  
plt.show()

sn.barplot(x='sex', y='Alcohol', data=new_df)  
plt.show()
# 남학생이 여학생에 비해서 음주 소비량이 높다.


# 2) Alcohol vs Pstatus
sn.countplot(x='Pstatus',  data=new_df) # 명목형 변수 빈도수
plt.show()

sn.barplot(x='Pstatus', y='Alcohol', data=new_df)  
plt.show()
# 부모거주자나 홀로 거주 모두 큰 차이 없음 


### 연속형(y) vs 이산형(x) 

# 1) Alcohol vs failures
sn.countplot(x='failures',  data=new_df) 
plt.show()

sn.barplot(x='failures', y='Alcohol', data=new_df)
plt.show()
# 낙제 과목수가 많을 수록 알콜 소비량이 높다.


# 2) Alcohol vs famrel
sn.barplot(x='famrel', y='Alcohol', data=new_df)
plt.show()
# 가족관계가 좋을 수록 알콜 소비량이 적다. 


### 연속형(x) vs 연속형(y) 

# 1) Alcohol vs age  
sn.scatterplot(x="age", y="Alcohol", data=new_df) 
plt.show()
# y축으로 분산 : 각 연령대별로 음주량이 분산

group = new_df.groupby('age')
result = group['Alcohol'].mean()
result

sn.scatterplot(x=result.index, y=result.values) 
plt.show()
# 전반적으로 나이가 많을 수록 음주량이 많아진다.


# 2) Alcohol vs grade  
sn.scatterplot(x="grade", y="Alcohol", data=new_df) 
plt.show()
# x축으로 분산 : 점수별로 음주량이 분산 

group = new_df.groupby('grade')
result = group['Alcohol'].mean()
result

sn.scatterplot(x=result.index, y=result.values) 
plt.show()
'''
0~30점 : 점진적으로 알콜 소비량 증가 
30~60점 : 점진적으로 알콜 소비량 감소 
'''
