'''
회귀분석용 sklearn dataset 정리 
'''
import sklearn 

# sklearn : 기계학습 관련 도구 패키지 
dir(sklearn)

# sklearn 모듈 확인 
print(sklearn.__all__)
'''
cluster : 군집모델 
datasets : 실습용 데이터셋  
ensemble : 앙상블모델 
linear_model : 선형회귀모델 
metrics : 모델 평가 
model_selection : train/test 나누기 
neural_network : 신경망 모델 
preprocessing : 전처리 
svm : SVM 모델 
tree : Tree 모델 
'''

from sklearn import datasets # dataset 제공 library

######################################
# 선형회귀분석에 적합한 데이터셋
######################################

# 1. 붓꽃(iris) : 회귀와 분류 모두 사용 
'''
붓꽃(iris) 데이터
- 붓꽃 데이터는 통계학자 피셔(R.A Fisher)의 붓꽃의 분류 연구에 기반한 데이터

• 타겟 변수 : y변수
세가지 붓꽃 종(species) : setosa, versicolor, virginica

•특징 변수(4) : x변수
꽃받침 길이(Sepal Length)
꽃받침 폭(Sepal Width)
꽃잎 길이(Petal Length)
꽃잎 폭(Petal Width)
'''
iris = datasets.load_iris() # dataset load 
print(iris) 
print(iris.DESCR) # dataset 설명제공 : 변수특징, 요약통계 

# X, y변수 선택 
iris_X = iris.data # x변수 
iris_y = iris.target # y변수

# 객체형과 모양확인 
print(type(iris_X))
print(type(iris_y))

print(iris_X.shape) # (150, 4) : 2d
print(iris_y.shape) # (150,) : 1d


# X변수명과 y변수 범주명 
print(iris.feature_names)# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target_names) # ['setosa' 'versicolor' 'virginica']


# DataFrame 변환  
import pandas as pd
iris_df = pd.DataFrame(iris_X, columns=iris.feature_names)

# y변수 추가 
iris_df['species'] = iris.target 
iris_df.head()
iris_df.info() 


# 차트 분석 : 각 특징별 타겟변수의 분포현황  
import seaborn as sn
import matplotlib.pyplot as plt

# 변수 간 산점도 : hue = 집단변수 : 집단별 색상 제공 
sn.pairplot(iris_df, hue="species")
plt.show() 


# 2. 당료병 데이터셋
'''
- 442명의 당뇨병 환자를 대상으로한 검사 결과를 나타내는 데이터

•타겟 변수 : y변수
1년 뒤 측정한 당료병 진행상태 정량적화 자료(연속형)

•특징 변수(10: 모두 정규화된 값) : x변수
age : 나이 (세)
sex : 성별 
bmi : 비만도지수
bp : 평균혈압(Average blood pressure)
S1 ~ S6: 기타 당료병에 영향을 미치는 요인들 
'''

diabetes = datasets.load_diabetes() # dataset load 
X = diabetes.data
y = diabetes.target 

print(diabetes.DESCR) # 컬럼 설명, url
'''
:Target: Column 11 -> 1년기준으로 질병 진행상태를 정량적(연속형)으로 측정 
:Attribute Information: Age ~ S6
'''    

print(diabetes.feature_names) # X변수명 
#print(diabetes.target_names) # None : 연속형 변수 이름 없음 

# X, y변수 동시 선택 
X, y = datasets.load_diabetes(return_X_y=True)

print(X.shape) # (442, 10) 
print(y.shape) # (442,) 



# 3. california 주택가격 
'''
•타겟 변수 : y변수
1990년 캘리포니아의 각 행정 구역 내 주택 가격의 중앙값

•특징 변수(8) : x변수
MedInc : 행정 구역 내 소득의 중앙값
HouseAge : 행정 구역 내 주택 연식의 중앙값
AveRooms : 평균 방 갯수
AveBedrms : 평균 침실 갯수
Population : 행정 구역 내 인구 수
AveOccup : 평균 자가 비율
Latitude : 해당 행정 구역의 위도
Longitude : 해당 행정 구역의 경도
'''
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
print(california.DESCR)

# X변수 -> DataFrame 변환 
cal_df = pd.DataFrame(california.data, columns=california.feature_names)
# y변수 추가 
cal_df["MEDV"] = california.target
cal_df.tail()
cal_df.info() 

############################################################################################################################

'''
분류분석용 sklearn dataset 정리
'''
from sklearn import datasets # dataset 제공 library
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

######################################
# 분류분석에 적합한 데이터셋
######################################

# 1. wine 
'''
와인의 화학 조성을 사용하여 와인의 종류를 예측하기 위한 데이터

•타겟 변수 : y변수 
◦와인의 종류 : 0, 1, 2 세가지 값 

•특징 변수 : x변수  
◦알콜(Alcohol)
◦말산(Malic acid)
◦회분(Ash)
◦회분의 알칼리도(Alcalinity of ash) 
◦마그네슘(Magnesium)
◦총 폴리페놀(Total phenols)
◦플라보노이드 폴리페놀(Flavanoids)
◦비 플라보노이드 폴리페놀(Nonflavanoid phenols)
◦프로안토시아닌(Proanthocyanins)
◦색상의 강도(Color intensity)
◦색상(Hue)
◦희석 와인의 OD280/OD315 비율 (OD280/OD315 of diluted wines)
◦프롤린(Proline)
'''
from sklearn.datasets import load_wine
wine = load_wine()
print(wine.target_names) # ['class_0', 'class_1', 'class_2']
print(wine.feature_names)


X, y = load_wine(return_X_y=True)
print(np.shape(X)) # (178, 13) : matrix
print(np.shape(y)) # (178,) : vector

# numpy -> DataFrame 
wine_df = pd.DataFrame(X, columns=wine.feature_names)
wine_df['class'] = y

# class별 주요변수 간 산점도 
sn.pairplot(vars=["alcohol", "alcalinity_of_ash", "total_phenols", "flavanoids"], 
             hue="class", data=wine_df)
plt.show()


# 2. breast cancer 데이터셋
'''
유방암(breast cancer) 진단 데이터 

•타겟 변수 
 - 종양이 양성(benign)인지 악성(malignant)인지를 판별
•특징 변수(30개) 
 - 유방암 진단 사진으로부터 측정한 종양(tumar)의 특징값
'''
cancer = datasets.load_breast_cancer()
print(cancer)
print(cancer.DESCR)

cancer_x = cancer.data
cancer_y = cancer.target
print(np.shape(cancer_x)) # (569, 30) : matrix
print(np.shape(cancer_y)) # (569,) : vector

cencar_df = pd.DataFrame(cancer_x, columns=cancer.feature_names)
cencar_df['class'] = cancer.target
cencar_df.tail()

# 타겟 변수 기준 주요변수 간 산점도 
sn.pairplot(vars=["worst radius", "worst texture", "worst perimeter", "worst area"], 
             hue="class", data=cencar_df)
plt.show()


# 3. digits 데이터셋 - 숫자 예측(0~9)
'''
숫자 필기 이미지 데이터

•타겟 변수 
 - 0 ~ 9 : 10진수 정수 
•특징 변수(64픽셀) 
 -0부터 9까지의 숫자를 손으로 쓴 이미지 데이터
 -각 이미지는 0부터 15까지의 16개 명암을 가지는 8x8=64픽셀 해상도의 흑백 이미지
'''
digits = datasets.load_digits()
print(digits.DESCR)

print(digits.data.shape) # (1797, 64)
print(digits.target.shape) # (1797,)
print(digits) # 8x8 image of integer pixels in the range 0..16

# 첫번째 이미지 픽셀, 정답 
img2d = digits.data[0].reshape(8,8)
plt.imshow(img2d) # 0 확인 
digits.target[0] # 0 정답 

   
# 4. news group 
'''
- 20개의 뉴스 그룹 문서 데이터(문서 분류 모델 예문으로 사용)

•타겟 변수 
◦문서가 속한 뉴스 그룹 : 20개 

•특징 변수 
◦문서 텍스트 : 18,846
'''

from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all') # 'train', 'test'
# Downloading 20news dataset.

print(newsgroups.DESCR)
'''
**Data Set Characteristics:**

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality               1
    Features                  text
    =================   ==========
'''

# data vs target
newsgroups.data # text
len(newsgroups.data) # 18846

newsgroups.target # array([10,  3, 17, ...,  3,  1,  7])
len(newsgroups.target) # 18846

# 뉴스 그룹 : 20개 이름 
newsgroups.target_names # ['alt.atheism', ... 'talk.religion.misc']

############################################################################################################################

"""
sklearn 패키지 
 - python 기계학습 관련 도구 제공 
"""

from sklearn.datasets import load_diabetes # dataset(당료병) 
from sklearn.linear_model import LinearRegression # model
from sklearn.model_selection import train_test_split # train/test  
from sklearn.metrics import mean_squared_error, r2_score # 평가 도구   


# 1. dataset load 
diabetes = load_diabetes() # 객체 반환 
dir(diabetes)  
'''
['DESCR', : 설명문 
 'data',  : X변수 
 'feature_names', : X변수 이름 
 'target', : y변수 
]
'''
X_names = diabetes.feature_names
X_names # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

X, y = load_diabetes(return_X_y = True) # X변수, y변수 반환 
X.shape # (442, 10)
y.shape # (442,)

# 2. X, y변수 특징 
X.mean(axis = 0) # -1.6638274468590581e-16
'''
[-1.44429466e-18,  2.54321451e-18, -2.25592546e-16, -4.85408596e-17,
       -1.42859580e-17,  3.89881064e-17, -6.02836031e-18, -1.78809958e-17,
        9.24348582e-17,  1.35176953e-17]
'''
X.min() # -0.137767225690012
X.max() # 0.198787989657293


# y변수 
y.mean() # 152.13348416289594
y.min() # 25.0
y.max() # 346.0


# 3. train_test_split : 70 vs 30 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size=0.3, 
                                    random_state=123) 
'''
test_size : 모델 평가셋 비율 
random_state : 시드값 지정 
'''
X_train.shape # (309, 10)
X_test.shape # (133, 10)


# 4. model 생성 : 훈련셋(train) 
lr = LinearRegression() # model object
model = lr.fit(X=X_train, y=y_train) # model 학습  
dir(model)
'''
coef_ : 회귀계수 
intercept_ : 절편 
predict() : y의 예측치  
score() : 예측 점수(결정계수 : r^2) 
'''
model.coef_ # 각 변수의 기울기(회귀계수) 
'''
[  10.45319644, -261.16273528,  538.85049356,  280.72085805,
   -855.24407564,  472.1969838 ,  166.53481397,  309.88981052,
    684.06085168,  102.3789942 ]
'''
model.intercept_ # y의 절편 
# 152.61082386550538


# 5. model 평가 : 평가셋(test) 
y_pred = model.predict(X=X_test) 
y_pred # y 예측치  
y_true = y_test # y 관측치(정답) 


# 1) 평균제곱오차(MSE) : 0의 수렴정도    
MSE = mean_squared_error(y_true, y_pred)
print('MSE =', MSE) # MSE = 2926.8196257936324


# 2) 결정계수 : 1의 수렴정도      
score = r2_score(y_true, y_pred)
print('r2 score =', score) 
# r2 score = 0.5078253552814805


# 3) score() 이용 : model 과적합 유무 
model.score(X=X_train, y=y_train) # 훈련셋 : 0.517498
model.score(X=X_test, y=y_test) # 평가셋 : 0.507825


############################################################################################################################

"""
sklearn 패키지 
 - python 기계학습 관련 도구 제공 
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.linear_model import LinearRegression # model
from sklearn.model_selection import train_test_split # split
from sklearn.metrics import mean_squared_error, r2_score # 평가도구 


##############################
### load_iris
##############################

# 1. dataset load 
iris = load_iris()
X, y = load_iris(return_X_y=True)
X.shape # (150, 4)


# 2. 변수 선택 
y = X[:,0] # 첫번째 변수 
X = X[:,1:] # 2~4번째 변수 
y.shape # (150,)

X.shape # (150, 3)

# 3. train/test split 
X_train, X_test, y_train,y_test = train_test_split(X, y, 
                 test_size=0.3, random_state=123)

X_train.shape # (105, 3)
X_test.shape # (45, 3)

# 4. model 생성 : 훈련셋 
model = LinearRegression().fit(X=X_train, y=y_train) 

# X변수 기울기 : 3개  
model.coef_ # [ 0.63924286,  0.75744562, -0.68796484]

# y 절편 
model.intercept_ # 1.8609363992411714


X[0] # [3.5, 1.4, 0.2]
X1 = 3.5
X2 = 1.4
X3 = 0.2

y = y[0] #  5.1

# 다중선형 회귀방정식
y_pred = 1.8609363992411714 + X1*0.63924286 + X2*0.75744562 + X3*-0.68796484
y_pred # 5.021117309241172

err = y - y_pred # 오차 
err # 0.07888269075882803
squared_err = err**2 # 오차제곱 : 패널티 
squared_err # 0.006222478901352893


# 5. model 평가
y_pred = model.predict(X=X_test)
y_true = y_test

# 1) MSE : 0수렴 정도  
MSE = mean_squared_error(y_true, y_pred)
print('MSE =', MSE) # MSE = 0.11633863200224709


# 2) 결정계수(R-제곱) : 1수렴 정도  
score = r2_score(y_true, y_pred)
print('r2 score =', score) # r2 score = 0.854680765745176

model.score(X_train, y_train) # 0.8581515699458578
model.score(X_test, y_test) # 0.854680765745176

# 3) 시각화 평가 
import matplotlib.pyplot as plt
plt.plot(y_pred, color='r', linestyle='--', label='pred')
plt.plot(y_true, color='b', linestyle='-', label='Y')
plt.legend(loc='best')
plt.show()

############################################################################################################################

"""
csv file 자료 + 회귀모델  
"""

import pandas as pd # csv file read
from sklearn.linear_model import LinearRegression # model 
from sklearn.metrics import mean_squared_error, r2_score # 평가도구 
from sklearn.model_selection import train_test_split # split


# 1. dataset load
path = r'C:\ITWILL\4_Python_ML\data' # file path 

iris = pd.read_csv(path + '/iris.csv')
print(iris.info())

# 2. 변수 선택 
cols = list(iris.columns)
cols # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']

y_col = cols.pop(2) # 3칼럼 추출 & 제외 
y_col # 'Petal.Length' 

x_cols = cols[:-1]
x_cols # ['Sepal.Length', 'Sepal.Width', 'Petal.Width']

X = iris[x_cols] 
y = iris[y_col] 

X.shape # (150, 3)
y.shape # (150,)


# 3. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


X_train.shape # (105, 3)
X_test.shape # (45, 3)


# 4. model 생성 : train set 
lr = LinearRegression()
model = lr.fit(X_train, y_train)


# 5. model 평가 : test set 
y_pred = model.predict(X = X_test)

mse = mean_squared_error(y_true = y_test, y_pred = y_pred)
print(mse) # 0.12397491396059163

score = r2_score(y_true = y_test, y_pred = y_pred)
print(score) # 0.9643540833169766


y_test[:5] # 정답 : pandas  
y_pred[:5] # 예측치 : numpy 


# 5개 정답 vs 예측치 상관계수 
import pandas as pd 

df = pd.DataFrame({'y_true' : y_test[:5].values, 
                   'y_pred': y_pred[:5]})
df
'''
   y_true    y_pred
0     4.9  4.915435
1     5.5  5.816851
2     5.6  5.792097
3     4.1  3.764201
4     1.4  1.113584
'''

# -1 < r < 1
r = df.y_true.corr(df.y_pred) # 0.9956708730399723

r2_score = r**2
r2_score # 0.9913604874201806

############################################################################################################################

"""
특징변수(X변수) 데이터변환(features scaling) : 이물질 제거 
 1. 특징변수(x변수) : 값의 크기(scale)에 따라 model 영향을 미치는 경우
      ex) 범죄율(-0.01~0.99), 주택가격(99~999)
   1) 표준화 : X변수를 대상으로 정규분포가 될 수 있도록 평균=0, 표준편차=1로 통일 시킴 
      -> 회귀모델, SVM 계열은 X변수가 정규분포라고 가정하에 학습이 진행되므로 표준화를 적용   
   2) 최소-최대 정규화 : 서로 다른 척도(값의 범위)를 갖는 X변수를 대상으로 최솟값=0, 최댓값=1로 통일 시킴 
      -> 트리모델 계열(회귀모델 계열이 아닌 경우)에서 서로 다른 척도를 갖는 경우 적용 

 2. 타깃변수(y변수) : 로그변환(log1p() 함수 이용 ) 
"""

import pandas as pd # dataset
from sklearn.model_selection import train_test_split # split 
from sklearn.linear_model import LinearRegression # model 
from sklearn.metrics import mean_squared_error, r2_score # 평가 

from sklearn.preprocessing import scale # 표준화(mu=0, st=1) 
from sklearn.preprocessing import minmax_scale # 정규화(0~1)
import numpy as np # 로그변환 + 난수 


###############################
### 특징변수(x변수) 데이터변환 
###############################
# 1. dataset load
path = r'C:\ITWILL\4_Python_ML\data'

# - 1978 보스턴 주택 가격에 미치는 요인을 기록한 데이터 
boston = pd.read_csv(path + '/BostonHousing.csv')
boston.info()
'''
 0   CRIM       506 non-null    float64 : 범죄율
 1   ZN         506 non-null    float64 : 25,000 평방피트를 초과 거주지역 비율
 2   INDUS      506 non-null    float64 : 비소매상업지역 면적 비율
 3   CHAS       506 non-null    int64   : 찰스강의 경계에 위치한 경우는 1, 아니면 0
 4   NOX        506 non-null    float64 : 일산화질소 농도
 5   RM         506 non-null    float64 : 주택당 방 수
 6   AGE        506 non-null    float64 : 1940년 이전에 건축된 주택의 비율
 7   DIS        506 non-null    float64 : 직업센터의 거리
 8   RAD        506 non-null    int64   : 방사형 고속도로까지의 거리 
 9   TAX        506 non-null    int64   : 재산세율
 10  PTRATIO    506 non-null    float64 : 학생/교사 비율
 11  B          506 non-null    float64 : 인구 중 흑인 비율
 12  LSTAT      506 non-null    float64 : 인구 중 하위 계층 비율
 13  MEDV       506 non-null    float64 : y변수 : 506개 타운의 주택가격(단위 1,000 달러)
 14  CAT. MEDV  506 non-null    int64   : 사용 안함(제외)
'''

X = boston.iloc[:, :13] # 독립변수 
X.shape # (506, 13)

y = boston.MEDV # 종속변수 
y.shape #(506,)

# x,y변수 스케일링 안됨 
X.mean() # 70.07396704469443
y.mean() # 22.532806324110677

# 2. scaling 함수 
def scaling(X, y, kind='none') : # (X, y, 유형)
    # x변수 스케일링  
    if kind == 'minmax_scale' :  
        X_trans = minmax_scale(X) # 1. 정규화
    elif kind == 'scale' : 
        X_trans = scale(X) # 2. 표준화 
    elif kind == 'log' :  
        X_trans = np.log1p(np.abs(X)) # 3. 로그변환
    else :
        X_trans = X # 4. 기본 
    
    # y변수 로그변환 
    if kind != 'none' :
        y = np.log1p(y)  # np.log(y+1) 
    
    # train/test split 
    X_train,X_test,y_train,y_test = train_test_split(
        X_trans, y, test_size = 30, random_state=1)   
    
    print(f"scaling 방법 : {kind}, X 평균 = {X_trans.mean()}")
    return X_train,X_test,y_train, y_test


X_train,X_test,y_train,y_test = scaling(X, y,'scale') # 원본 자료 이용 

X_train.mean() # 0.3868069035401226
y_train.mean() # 3.0858307104922083

# 3. 회귀모델 생성하기
model = LinearRegression().fit(X=X_train, y=y_train) # 지도학습 


# 4. model 평가하기
model_train_score = model.score(X_train, y_train) 
model_test_score = model.score(X_test, y_test)
print('model train score =', model_train_score)
print('model test score =', model_test_score)
'''
model train score = 0.7410721208614652
model test score = 0.717046343087048
'''

y_pred = model.predict(X_test)
y_true = y_test
print('R2 score =',r2_score(y_true, y_pred)) # model test score 동일 
mse = mean_squared_error(y_true, y_pred)
print('MSE =', mse)

'''
1. 원형자료 : X, y변수 스케일링 전 
R2 score = 0.717046343087048
MSE = 20.20083182974836

2. X변수 정규화, y변수 로그변환 
R2 score = 0.7633961405434471
MSE = 0.027922682660046293

3. X변수 표준화, y변수 로그변환
R2 score = 0.763396140543447
MSE = 0.02792268266004631
'''

############################################################################################################################

"""
 가변수(dummy) 변환 : 명목형(범주형) 변수를 X변수 사용
"""

import pandas as pd # csv file, 가변수 
from sklearn.model_selection import train_test_split # split 
from sklearn.linear_model import LinearRegression # model 


# 1. csv file load 
path = r'C:\ITWILL\4_Python_ML\data'
insurance = pd.read_csv(path + '/insurance.csv')
insurance.info()



# 2. 불필요한 칼럼 제거 : region
new_df = insurance.drop(['region'], axis= 1)
new_df.info()
'''
 0   age      1338 non-null   int64  
 1   sex      1338 non-null   object  -> 가변수  
 2   bmi      1338 non-null   float64
 3   children  1338 non-null   int64   
 4   smoker   1338 non-null   object  -> 가변수 
 5   charges  1338 non-null   float64 -> y변수 
'''



# 3. X, y변수 선택 
X = new_df.drop('charges', axis= 1)
X.shape #  (1338, 5)
X.info()

y = new_df['charges'] # 의료비 


# 4. 명목형(범주형) 변수 -> 가변수(dummy) 변환 : k-1개 
X.info()
X_dummy = pd.get_dummies(X, columns=['sex', 'smoker'],
               drop_first=True, dtype='uint8')

X_dummy.info()


# 5. 이상치 확인  
X_dummy.describe() 
'''
age : 최댓값 이상치  
bmi : 최솟값 이상치 
'''
X_dummy.shape # (1338, 5)

# age 이상치 확인 
X_dummy[~((X_dummy.age > 0) & (X_dummy.age <= 100))].index
# age 이상치 index : [12, 114, 180]

# age 이상치 제거 
X_new = X_dummy[(X_dummy.age > 0) & (X_dummy.age <= 100)] # age기준 

# bmi 이상치 확인 
X_dummy[X_dummy.bmi < 0].index  
# bmi 이상치 index : [16, 48, 82]

# bmi 이상치 처리 
X_new = X_new[X_new.bmi > 0] # bmi 기준 
X_new.shape # (1332, 5)

X_new.index 

# y변수 정상범주 선정 
y = y[X_new.index] # X변수 색인 이용 
y.shape # (1332,)

# 6. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.3, random_state=123)


# 7. model 생성 & 평가 
model = LinearRegression().fit(X=X_train, y=y_train)

# model 평가 
model.score(X=X_train, y=y_train)
model.score(X=X_test, y=y_test)
'''
1. 이상치 처리 전 
0.6837522970202745
0.7236336310934985 -> r2 score 

2. 이상치 처리 후 
0.7425695212639727
0.7612276881341357 -> r2 score 
'''

