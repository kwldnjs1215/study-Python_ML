"""
로지스틱회귀모델을 이해를 위한 오즈비(odds_ratio)와 로짓변환 그리고 시그모이드함수    
 - 오즈비(odds_ratio) : 실패에 대한 성공확률 = p / 1-p
 - 로짓값(logit value) : 오즈비를 로그변환한 값 = log(오즈비) 
 - 시그모이드(sigmoid) : 로짓값을 0 ~ 1 사이 확률로 변환하는 함수      
"""

import numpy as np


# sigmoid 함수 정의 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# [실습] 오즈비(odds_ratio)와 로짓값 그리고 sigmoid함수 

# 1) 성공확률 50% 미만
p = 0.2   

odds_ratio = p / (1-p) # 오즈비(odds_ratio)=0.25 
# 오즈비(odds_ratio) < 1 이면 x가 감소 방향으로 y에 영향 

logit = np.log(odds_ratio) # 로짓값= -1.38629(음수 ~ -int)  
sig = sigmoid(logit) # sigmoid함수 = 0.2 
y_pred = 1 if sig > 0.5 else 0 # y예측치 = 0

# 2) 성공확률 50% 이상
p = 0.6 
  
odds_ratio = p / (1-p) # 오즈비(odds_ratio)=1.49999
# 오즈비(odds_ratio) > 1 이면 x가 증가 방향으로 y에 영향

logit = np.log(odds_ratio) # 로짓값=0.405(양수 ~ +inf)   
sig = sigmoid(logit) # sigmoid함수 = 0.6
y_pred = 1 if sig > 0.5 else 0 # y예측치 = 1


###########################################
### 통계적인 방법의 로지스틱회귀모델 
###########################################

import pandas as pd


path = r'C:\ITWILL\4_Python_ML\data'

skin = pd.read_csv(path + '/skin.csv')
skin.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   cust_no      30 non-null     int64  : 제외 
 1   gender       30 non-null     object : x변수
 2   age          30 non-null     int64 
 3   job          30 non-null     object
 4   marry        30 non-null     object
 5   car          30 non-null     object
 6   cupon_react  30 non-null     object : y변수
'''

# 1. X, y변수 인코딩
X = skin.drop(['cust_no','cupon_react'], axis = 1)  

# X변수 인코딩 : 2진수  
X = pd.get_dummies(X, columns=['gender', 'job' , 'marry', 'car'],
                   drop_first=True, dtype='uint8') 

# y변수 인코딩 : 10진수 
from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(skin.cupon_react)

new_skin = X.copy() 
new_skin['y'] = y 
new_skin.info()
'''
 0   age          30 non-null     int64
 1   gender_male  30 non-null     uint8
 2   job_YES      30 non-null     uint8
 3   marry_YES    30 non-null     uint8
 4   car_YES      30 non-null     uint8
 5   y            30 non-null     int32
'''


# 2. 상관계수 행렬 : |0.25| 미만 변수 제외 
corr = new_skin.corr()
corr['y']
'''
age            0.276329
gender_male   -0.302079
job_YES        0.221719 -> 제외 
marry_YES      0.475651
car_YES        0.185520 -> 제외
y              1.000000
'''


# 3. 로지스틱회귀모델 : formula 형식 
from statsmodels.formula.api import logit

formula = logit(formula='y ~ age + gender_male + marry_YES', data = new_skin)
model = formula.fit()

dir(model)
'''
fittedvalues : 적합치 
params : 회귀계수 
summary() : 분석결과 
'''
y = new_skin.y # 종속변수(0 or 1) 
y_fitted = model.fittedvalues # model 적합치(예측치)=로짓값      

# 로짓값 -> 확률(sigmoid func)
y_sig = sigmoid(y_fitted)
y_sig

# 확률 -> 0 or 1 변환 
y_pred = [ 1 if y > 0.5 else 0 for y in y_sig]
y_pred

result = pd.DataFrame({'y' : y,'y_sig' :y_sig,'y_pred':y_pred})
print(result)

##################################################################################################################################

"""
 - 로지스틱회귀모델 & ROC 평가 
"""

from sklearn.datasets import load_breast_cancer # dataset
from sklearn.linear_model import LogisticRegression # model 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import confusion_matrix, accuracy_score # model 평가 


################################
### 이항분류(binary class) 
################################

# 1. dataset loading 
X, y = load_breast_cancer(return_X_y=True)

print(X.shape) # (569, 30)
print(y) # 0(B) or 1(M)


# 2. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=1)


# 3. model 생성 
lr = LogisticRegression(solver='lbfgs', max_iter=100, random_state=1)  
'''
solver='lbfgs', : 최적화에 사용되는 기본 알고리즘(solver) 
max_iter=100,  : 반복학습횟수 
random_state=None, : 난수 seed값 지정 
'''

model = lr.fit(X=X_train, y=y_train) 
dir(model)
'''
predict() : y 클래스 예측 
predict_proba() : y 확률 예측(sigmoid 함수) 
'''

# 4. model 평가 
y_pred = model.predict(X = X_test) # class 예측치 
y_pred_proba = model.predict_proba(X = X_test) # 확률 예측치 

y_true = y_test # 관측치 

# 1) 혼동행렬(confusion_matrix)
con_max = confusion_matrix(y_true, y_pred)
print(con_max)
'''
     0    1
0 [[ 59   4]
1 [  5 103]]
'''

# 2) 분류정확도 
acc = accuracy_score(y_true, y_pred)
print('accuracy =',acc) # accuracy = 0.9473684210526315

(59 + 103) / len(y_pred) # 0.9473684210526315


#############################
# ROC curve 시각화
#############################

# 1) 확률 예측치
y_pred_proba = model.predict_proba(X = X_test) # 확률 예측 
y_pred_proba = y_pred_proba[:, 1] # 악성(pos) 확률 추출   


# 2) ROC curve 
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 

fpr, tpr, _ = roc_curve(y_true, y_pred_proba) #(실제값, 1예측확률)
'''
 x축 : FPR = 1 - TNR(특이도) 
 y축 : TPR(민감도) 
'''

plt.plot(fpr, tpr, color = 'red', label='ROC curve')
plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='AUC')
plt.legend()
plt.show()

'''
ROC curve FPR vs TPR  

ROC curve x축 : FPR(False Positive Rate) - 실제 음성을 양성으로 잘못 예측할 비율  
ROC curve y축 : TPR(True Positive Rate) - 실제 양성을 양성으로 정상 예측할 비율  
'''

'''
        0(n)   1(p)
0(n) [[ 59(TN)   4]   = 63
1(p)  [  5  103(TP)]] = 108
'''
TPR = 103 / 108 # 0.9537037037037037 : 민감도(Y축)
TNR = 59 / 63 # 0.9365079365079365 : 특이도
FPR = 1 - TNR  # 0.06349206349206349 : 위양성비율(X축)

##################################################################################################################################

"""
 - 다항분류기(multi class classifier)  
"""
from sklearn.datasets import load_digits # dataset
from sklearn.linear_model import LogisticRegression # model 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import confusion_matrix, accuracy_score # model 평가 


# 1. dataset loading 
digits = load_digits()

image = digits.data # x변수 
label = digits.target # y변수 

image.shape # (1797, 64) : (size, pixel) 
image[0] # 0 ~ 15 
label.shape 
label # 0 ~ 9
 
# 2. train_test_split
img_train, img_test, lab_train, lab_test = train_test_split(
                 image, label, 
                 test_size=0.3, 
                 random_state=123)


# 3. model 생성 
lr = LogisticRegression(random_state=123,
                   solver='lbfgs',
                   max_iter=100, 
                   multi_class='multinomial')
'''
penalty : {'l1', 'l2', 'elasticnet', None}, default='l2' 
  -> 과적합 규제 : 'l1' - lasso 회귀 , 'l2' - ridge 회귀 
C : float, default=1.0
  -> Cost(비용함수)  
random_state : int, RandomState instance, default=None
  -> 시드값 
solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, \
        default='lbfgs'
  -> 최적화 알고리즘       
max_iter : int, default=100  
  -> 반복횟수       
multi_class='auto' : 다항분류(multinomial) 
'''

model = lr.fit(X=img_train, y=lab_train)


# 4. model 평가 
y_pred = model.predict(img_test) # class 예측 
y_pred_proba = model.predict_proba(img_test) # 확률 예측 

y_pred_proba.shape # (540, 10)
y_pred_proba[0].sum() # 1

# 1) 혼동행렬(confusion matrix)
con_mat = confusion_matrix(lab_test, y_pred)
print(con_mat)

# 2) 분류정확도(Accuracy)
accuracy = accuracy_score(lab_test, y_pred)
print('Accuracy =', accuracy) 
# Accuracy = 0.9666666666666667


# 3) heatmap 시각화 
import matplotlib.pyplot as plt
import seaborn as sn
  
# confusion matrix heatmap 
plt.figure(figsize=(6,6)) # size
sn.heatmap(con_mat, annot=True, fmt=".3f",
           linewidths=.5, square = True) 
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: ', format(accuracy,'.6f')
plt.title(all_sample_title, size = 18)
plt.show()

# Accuracy = 0.9666666666666667

##################################################################################################################################

"""
주성분 분석(PCA : Principal Component Analysis)
 1. 다중공선성의 진단 :  다중회귀분석모델 문제점 발생  
 2. 차원 축소 : 특징 수를 줄여서 다중공선성 문제 해결 
"""

from sklearn.decomposition import PCA # 주성분 분석 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols # logit
import pandas as pd 

  
# 1.iris dataset load      
iris = load_iris()

X = iris.data
y = iris.target
'''
array([[1. , 5.1, 3.5, 1.4, 0.2],
       [1. , 4.9, 3. , 1.4, 0.2],
       [1. , 4.7, 3.2, 1.3, 0.2],
       [1. , 4.6, 3.1, 1.5, 0.2],
       [1. , 5. , 3.6, 1.4, 0.2],
'''       

df = pd.DataFrame(X, columns= ['x1', 'x2', 'x3', 'x4'])
corr = df.corr()
print(corr)

df['y'] = y 
df.columns  # ['x1', 'x2', 'x3', 'x4', 'y']


# 2. 다중선형회귀분석 
ols_obj = ols(formula='y ~ x1 + x2 + x3 + x4', data = df)
model = ols_obj.fit()

dir(model)
'''
params : 절편과 회귀계수 
fittedvalues : y의 적합치 
summary() : 분석결과 
'''

model.params
'''
Intercept    0.186495 : 절편 
x1          -0.111906 : 회귀계수(기울기)
x2          -0.040079
x3           0.228645
x4           0.609252
'''

# 회귀분석 결과 제공  
print(model.summary()) 
'''
Model:                            OLS   Adj. R-squared:                  0.928
Method:                 Least Squares   F-statistic:                     484.5
Date:                Thu, 05 May 2022   Prob (F-statistic):           8.46e-83
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.1865      0.205      0.910      0.364      -0.218       0.591
x1            -0.1119      0.058     -1.941      0.054      -0.226       0.002
x2            -0.0401      0.060     -0.671      0.503      -0.158       0.078
x3             0.2286      0.057      4.022      0.000       0.116       0.341
x4             0.6093      0.094      6.450      0.000       0.423       0.796
==============================================================================
Prob (F-statistic): 모델의 통계적 유의성 
Adj. R-squared: 모델의 설명력 
coef : 절편 & 회귀계수 
std err : 표준오차는 회귀계수(coef)의 추정치의 정확성
t : t 검정 통계량 = (표본평균-모평균) / (표본표준편차/sqrt(표본수)) 
P>|t| : 유의확률(t 검정 통계량 근거) 5% 기준으로 가설 채택/기각 결정
    5% 미만인 경우 해당 독립변수는 종속변수에 영향이 있다라고 할 수 있다.   
'''


#  3. 다중공선성의 진단
'''
분산팽창요인(VIF, Variance Inflation Factor) : 다중공선성 진단  
통상적으로 10보다 크면 다중공선성이 있다고 판단
''' 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 형식) variance_inflation_factor(exog, exog_idx)
dir(ols_obj)
'''
exog : 독립변수 
endog : 종속변수 
'''
exog = ols_obj.exog # 엑소(exog) : 모델에서 사용되는 독립변수 
   
# 다중공선성 진단  
for idx in range(1,5) : # 1~4
    print(variance_inflation_factor(exog, idx)) # idx=1~4
    
'''
7.072722013939533
2.1008716761242523
31.26149777492164
16.090175419908462
'''

df.iloc[:,:4].corr()

    
# 4. 주성분분석(PCA)

# 1) 주성분분석 모델 생성 
pca = PCA() # random_state=123
X_pca = pca.fit_transform(X)
print(X_pca)

X_pca.shape # (150, 4)

# 2) 고유값이 설명가능한 분산비율(분산량)
var_ratio = pca.explained_variance_ratio_ # 85% 이상 -> 95% 이상 권장  
print(var_ratio) # [0.92461872 0.05306648 0.01710261 0.00521218]

# 제1주성분 + 제2주성분 
sum(var_ratio[:2]) # 0.977685206318795 = 98%(2% 손실)

# 3) 스크리 플롯 : 주성분 개수를 선택할 수 있는 그래프(Elbow Point : 완만해지기 이전 선택)
plt.bar(x = range(4), height=var_ratio)
plt.plot(var_ratio, color='r', linestyle='--', marker='o') ## 선 그래프 출력
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.xticks(range(4), labels = range(1,5))
plt.show()


# 4) 주성분 결정 : 분산비율(분산량) 95%에 해당하는 지점
print(X_pca[:, :2]) # 주성분분석 2개 차원 선정  

X_new = X_pca[:, :2]

X_new.shape # (150, 2) 
print(X_new)

# 5. 주성분분석 결과를 회귀분석과 분류분석의 독립변수 이용 

from sklearn.linear_model import LinearRegression # 선형회귀모델  
from sklearn.linear_model import LogisticRegression # 로지스틱회귀모델  

##################################
# LinearRegression : X vs X_new
##################################

# 원형 자료 
lr_model1 = LinearRegression().fit(X = X, y = y)
lr_model1.score(X = X, y = y) # r2 score 
# 0.9303939218549564

# 주성분 자료 
lr_model2 = LinearRegression().fit(X = X_new, y = y)
lr_model2.score(X = X_new, y = y) # r2 score
# 0.9087681620170027

##################################
# LogisticRegression : X vs X_new
##################################

# 원형 자료 
lr_model1 = LogisticRegression().fit(X = X, y = y)
lr_model1.score(X = X, y = y) # accuracy
# 0.9733333333333334

# 주성분 자료 
lr_model2 = LogisticRegression().fit(X = X_new, y = y)
lr_model2.score(X = X_new, y = y) # accuracy
# 0.9666666666666667
