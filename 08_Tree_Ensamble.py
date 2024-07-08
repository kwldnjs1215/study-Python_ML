'''
Decision Tree 모델 & 시각화 
"""

'''
import pandas as pd # csv file read 
from sklearn.tree import DecisionTreeClassifier # 의사결정트리 모델  
from sklearn.metrics import accuracy_score # model 평가 

# tree 시각화 
from sklearn.tree import plot_tree, export_graphviz
from graphviz import Source # 설치 필요 : pip install graphviz

# 1. dataset load 
path = r'c:\ITWILL\4_Python_ML\data'
dataset = pd.read_csv(path + "/tree_data.csv")
print(dataset.info())
'''
iq         6 non-null int64 - iq수치
age        6 non-null int64 - 나이
income     6 non-null int64 - 수입
owner      6 non-null int64 - 사업가 유무
unidegree  6 non-null int64 - 학위 유무
smoking    6 non-null int64 - 흡연 유무 - y변수 
'''

# 2. 변수 선택 
cols = list(dataset.columns)
X = dataset[cols[:-1]]
y = dataset[cols[-1]] # 0 or 1 

# 3. model & 평가 
model = DecisionTreeClassifier(random_state=123).fit(X, y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
print(acc) # 1.0 


# 4. tree 시각화 
feature_names = cols[:-1]  # x변수 이름 

# 의사결정트리 시각화 
plot_tree(model, feature_names = feature_names)  

# y변수 class 이름 
class_names = ['no', 'yes'] # 0 or 1


# 의사결정트리 파일 저장 & 콘솔 시각화 
graph = export_graphviz(model,
                out_file="tree_graph.dot",
                feature_names = feature_names,
                class_names = class_names,
                rounded=False,
                impurity=True, # 지니계수/엔트로피 
                filled=True)

#  file load 
file = open("tree_graph.dot", mode = 'r') 
dot_graph = file.read()
  
Source(dot_graph) # tree 시각화 : Console 출력
# 오류 발생 시 : spyder 재시작 

'''
분류결과 해석
비흡연자 : 수입 적음 
흡연자 : 수입 많고, 지능지수가 높음 
지능지수 105 일때 
  상대적인 수입이 작은 흡연자 
  상대적인 수입이 많은 비흡연자 
'''

###################################################################################################

'''
의사결정트리 주요 Hyper parameter 
'''
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# tree 시각화 
from sklearn.tree import export_graphviz
from graphviz import Source  


iris = load_iris()
x_names = iris.feature_names # x변수 이름 
x_names
'''
['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']
'''

labels = iris.target_names # ['setosa', 'versicolor', 'virginica']

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


############################
### Hyper parameter 
############################
'''
criterion='gini' : 중요변수 선정 기준, 
 -> criterion : {"gini", "entropy"}, default="gini"
splitter='best' : 각 노드에서 분할을 선택하는 데 사용되는 전략, 
max_depth=None : tree 최대 깊이, 
 -> max_depth : int, default=None
 -> max_depth=None : min_samples_split의 샘플 수 보다 적을 때 까지 tree 깊이 생성 
 -> 과적합 제어 역할 : 값이 클 수록 과대적합, 적을 수록 과소적합 
min_samples_split=2 : 내부 노드를 분할하는 데 필요한 최소 샘플 수(기본 2개)
 -> int or float, default=2    
 -> 과적합 제어 역할 : 값이 클 수록 과소적합, 적을 수록 과대적합 
'''

# model : default parameter
model = DecisionTreeClassifier(criterion='gini',
                               random_state=123, 
                               max_depth=None,
                               min_samples_split=2)

model.fit(X=X_train, y=y_train) # model 학습 

dir(model)
'''
feature_importances_ : x변수의 중요도 
get_depth() : 트리 깊이 
max_features_ : x변수 최대 길이 
score() : 분류정확도 
'''

model.get_depth() # 5
model.max_features_ #  4
model.feature_importances_ 
# [0.01364196, 0.01435996, 0.5461181 , 0.42587999]

# model 평가 : 과적합(overfitting) 유무 확인  
model.score(X=X_train, y=y_train) # 1.0
model.score(X=X_test, y=y_test) # 0.9555555555555556


# tree 시각화 
graph = export_graphviz(model,
                out_file="tree_graph.dot",
                feature_names=x_names,
                class_names=labels,
                rounded=True,
                impurity=True,
                filled=True)


# dot file load 
file = open("tree_graph.dot") 
dot_graph = file.read()

# tree 시각화 : Console 출력  
Source(dot_graph) 

'''
분류결과 해석 
setosa : 꽃잎길이(3번)가 작은 경우 
virginica : 꽃잎길이(3번)와 꽃잎넓이(4번) 큰 경우 
vircolor : 꽃잎길이(3번)와 꽃잎넓이(4번) 중간 정도 큰 경우 
'''

##################################
#### 트리 깊이 지정 : 과적합 해결 
##################################
model2 = DecisionTreeClassifier(criterion='entropy',
                               random_state=123, 
                               max_depth=3)

model2.fit(X=X_train, y=y_train) # model 학습 


model2.score(X=X_train, y=y_train) # 0.9809523809523809
model2.score(X=X_test, y=y_test) # 0.9333333333333333

# tree 시각화 
graph = export_graphviz(model2,
                out_file="tree_graph.dot",
                feature_names=x_names,
                class_names=labels,
                rounded=True,
                impurity=True,
                filled=True)


# dot file load 
file = open("tree_graph.dot") 
dot_graph = file.read()

# tree 시각화 : Console 출력  
Source(dot_graph) 

###################################################################################################

"""
 Label Encoding
 - 일반적으로 y변수(대상변수)를 대상으로 인코딩 
 - 트리 계열 모델(의사결정트리, 랜덤포레스트)에서 x변수에 적용
"""

import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder # 인코딩 도구(10진수) 
import matplotlib.pyplot as plt # 중요변수 시각화 

# 1. 화장품 데이터(skin.csv) 가져오기 
df = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\skin.csv")
df.info()
'''
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   cust_no      30 non-null     int64  -> 제외 
 1   gender       30 non-null     object -> x변수  
 2   age          30 non-null     int64  -> x변수 
 3   job          30 non-null     object -> x변수
 4   marry        30 non-null     object -> x변수
 5   car          30 non-null     object -> x변수
 6   cupon_react  30 non-null     object -> y변수(화장품 구입여부) 
'''
   

# 범주형 변수의 범주(category) 확인 
def category_view(df, cols) : 
    for name in cols :
        print('{0} -> {1}'.format(name, df[name].unique()))


 
category_view(df, df.columns)


# 2. X, y변수 선택 
X = df.drop(['cust_no', 'cupon_react'], axis = 1) # X변수 선택 
y = df['cupon_react'] # y변수 선택 


# 3. data 인코딩 : 문자형 -> 숫자형 

# X변수 인코딩 
X['gender'] = LabelEncoder().fit_transform(X['gender'])
X['job'] = LabelEncoder().fit_transform(X['job'])
X['marry'] = LabelEncoder().fit_transform(X['marry'])
X['car'] = LabelEncoder().fit_transform(X['car'])

X.info()


# y변수 인코딩
y = LabelEncoder().fit_transform(y)  

                                                                                     
# 4.훈련 데이터 75, 테스트 데이터 25으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state = 123)
'''
test_size = 0.25 생략 : Train : 75% -> Test : 25%
test_size = 0.25 작성 : Test : 25% -> Train : 75%
'''

# 5. model 생성 : DecisionTree 분류기 
model = DecisionTreeClassifier().fit(X_train, y_train) # ValueError
# 인코딩 생략 시 오류 발생(ValueError)


# 6. 중요 변수 
print("중요도 : \n{}".format(model.feature_importances_))
# 중요도 : [0.00691824 0.3211478  0.34591195 0.2132015  0.11282051]


x_size = 5 # x변수 개수
x_names = list(X.columns) # x변수명 추출 
 
# 중요변수 시각화 : 가로막대 차트 
plt.barh(range(x_size), model.feature_importances_) 
plt.yticks(range(x_size), x_names) # y축 눈금 : x변수명 적용  
plt.xlabel('feature_importances')
plt.show()


# 7. 모델 평가  
y_pred= model.predict(X_test) # 예측치

# 분류정확도 
accuracy = accuracy_score( y_test, y_pred)
print( accuracy) # 0.875 

# 혼동행렬
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)    
'''
    0 1 
0 [[4 0]
1  [1 3]]
'''


precision = 4/5 # 0.8 
recall = 3/4 #  0.75

# 정밀도, 재현율, f1 score 확인 
report = classification_report(y_test, y_pred)
print(report)
'''
              precision    recall  f1-score   support

           0       0.80      1.00      0.89         4
           1       1.00      0.75      0.86         4

    accuracy                           0.88         8
   macro avg       0.90      0.88      0.87         8
weighted avg       0.90      0.88      0.87         8

macro avg = 각 평가에 대한 산술평균 
               precision : 0.9 = (0.80 + 1.00) / 2
weighted avg = support를 가중치로 한 가중평균 
               precision : 0.90 = ((0.80 * 4) + (1.00 * 4)) / 8
               f1-score : 0.875 = ((0.89 * 4) + (0.86 * 4)) / 8
'''


###################################################################################################


"""
지니불순도(Gini-impurity), 엔트로피(Entropy)
  Tree model에서 중요변수 선정 기준
 확률 변수 간의 불확실성을 나타내는 수치
 무질서의 양의 척도, 작을 수록 불확실성이 낮다.

  지니불순도와 엔트로피 수식  
 Gini-impurity = sum(p * (1-p))
 Entropy = -sum(p * log(p))

  지니계수와 정보이득  
  gini_index = base - Gini-impurity # 0.72
  info_gain = base - Entropy
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier # model 
from sklearn.metrics import confusion_matrix # 평가 
# 시각화 도구 
from sklearn.tree import plot_tree, export_graphviz 
from graphviz import Source 

##########################
### dataset 적용 
##########################

# 1. data set 생성 함수
def createDataSet():
    dataSet = [[1, 1, 'yes'],
    [1, 1, 'yes'],
    [1, 0, 'no'],
    [0, 1, 'no'],
    [0, 1, 'no']]
    columns = ['dark_clouds','gust'] # X1,X2,label
    return dataSet, columns


# 함수 호출 
dataSet, columns = createDataSet()

# list -> numpy 
dataSet = np.array(dataSet)
dataSet.shape # (5, 3)
print(dataSet)
print(columns) # ['dark_clouds', 'gust']

# 변수 선택 
X = dataSet[:, :2]
y = dataSet[:, -1]

# 레이블 인코딩 : 'yes' = 1 or 'no' = 0
label = [1 if i == 'yes' else 0 for i in y] 


# model 생성 
obj = DecisionTreeClassifier(criterion='entropy')
model = obj.fit(X = X, y = label)

y_pred = model.predict(X)

# 혼동행렬 
con_mat = confusion_matrix(label, y_pred)
print(con_mat)

# tree 시각화 : Plots 출력  
plot_tree(model, feature_names=columns)

# tree graph 
export_graphviz(decision_tree=model, 
                out_file='tree_graph.dot',
                max_depth=3,
                feature_names=columns,
                class_names=True)

# # dot(화소) file load 
file = open("tree_graph.dot")
dot_graph = file.read()

# tree 시각화 : Console 출력  
Source(dot_graph)


###################################################################################################

"""
RandomForest 앙상블 모델 
"""

from sklearn.ensemble import RandomForestClassifier # model(분류기)
from sklearn.model_selection import train_test_split # dataset split  
from sklearn.datasets import load_wine # dataset 

# 평가 도구 
from sklearn.metrics import confusion_matrix, classification_report

# 1. dataset load
wine = load_wine()

X, y = wine.data, wine.target
X.shape # (178, 13)
y # 0 ~ 2

#  2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=123) # 90% 훈련셋 


# 3. model 생성 
'''
주요 hyper parameter(default)
 n_estimators=100 : tree 개수 
 criterion='gini' : 중요변수 선정 기준 
 max_depth=None : 트리 깊이 
 min_samples_split=2 : 내부 노드 분할에 필요한 최소 샘플 수
'''

# 훈련셋 적용 : 전체데이터셋 적용 
model = RandomForestClassifier(random_state=123).fit(X = X, y = y) 



# 4. model 평가 
y_pred = model.predict(X = X_test) # new data 


# 혼동행렬
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)


# 분류 리포트 
print(classification_report(y_test, y_pred))


# 5. 중요변수 시각화 
print('중요도 : ', model.feature_importances_)
'''
 중요도 :  [0.10968347 0.0327231  0.0122728  0.02942002 0.02879057 0.05790394
 0.13386676 0.00822129 0.02703471 0.158535   0.07640667 0.13842681
 0.18671486]
'''
x_names = wine.feature_names # x변수 이름  
x_size = len(x_names) # x변수 개수  

import matplotlib.pyplot as plt 

# 가로막대 차트 
plt.barh(range(x_size), model.feature_importances_) # (y, x)
plt.yticks(range(x_size), x_names)   
plt.xlabel('feature_importances') 
plt.show()


###################################################################################################

'''
k겹 교차검정(cross validation)
 - 전체 dataset을 k등분 
 - 검정셋과 훈련셋을 서로 교차하여 검정하는 방식 
'''

from sklearn.datasets import load_digits # 0~9 손글씨 이미지 
from sklearn.ensemble import RandomForestClassifier # RM model 
from sklearn.metrics import accuracy_score # 평가 
from sklearn.model_selection import train_test_split # 홀드아웃 검증 
from sklearn.model_selection import cross_validate # 교차검증  
import numpy as np # Testset 선정

# 1. dataset load 
digits = load_digits()

X = digits.data
y = digits.target

X.shape # (1797, 64) 
y.shape # (1797,)

# 2. model 생성 : tree 100개 학습 
model = RandomForestClassifier().fit(X, y) # full dataset 이용 


# 3. Test set 선정 : 500개 이미지  
idx = np.random.choice(a=len(X), size=500, replace = False)
X_test = X[idx]
y_test = y[idx]



# 4. 평가셋 이용 : model 평가(1회) 
y_pred = model.predict(X = X_test) # 예측치  
y_pred 

accuracy = accuracy_score(y_test, y_pred)
print(accuracy) # 1.0


# 5. k겹 교차검정 이용 : model 평가(5회)  
score = cross_validate(model, X_test, y_test, cv=5)
print(score)

# 산술평균으로 model 성능 결정 
score['test_score'].mean() # 0.9400000000000001
# array([0.92, 0.95, 0.94, 0.94, 0.95])

###################################################################################################

"""
 1. RandomForest Hyper parameters
 2. GridSearch : best parameters 
"""

from sklearn.ensemble import RandomForestClassifier # model 
from sklearn.datasets import load_digits # dataset 
from sklearn.model_selection import train_test_split # dataset split 

# 1. dataset load
X, y = load_digits(return_X_y=True)


#  2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=15) # 90% 훈련셋 


# 3. 기본 model 생성 
#help(RandomForestClassifier)
model = RandomForestClassifier(random_state=234) # default 적용 
'''
주요 hyper parameter(default) 
n_estimators=100 : 결정 트리 개수, 많을 수록 성능이 좋아짐
criterion='gini' : 중요변수 선정기준 : {"gini", "entropy"}
max_depth=None : min_samples_split의 샘플 수 보다 적을 때 까지 tree 깊이 생성
min_samples_split=2 : 내부 node 분할에 사용할 최소 sample 개수
max_features='auto' : 최대 사용할 x변수 개수 : {"auto", "sqrt", "log2"}
min_samples_leaf=1 : leaf node를 만드는데 필요한 최소한의 sample 개수
n_jobs=None : cpu 사용 개수
'''

# model 학습 
model.fit(X = X_train, y = y_train) # 90% : full datast 

# model 평가 
model.score(X = X_test, y = y_test) # 10% : new dateset 


# 3.model tuning : GridSearch model
from sklearn.model_selection import GridSearchCV # best parameters


parmas = {'n_estimators' : [100, 150, 200],
          'max_depth' : [None, 3, 5, 7],
          'max_features' : ["auto", "sqrt"],
          'min_samples_split' : [2, 10, 20],
          'min_samples_leaf' : [1, 10, 20]} # dict 정의 

grid_model = GridSearchCV(model, param_grid=parmas, 
                          scoring='accuracy',cv=5, n_jobs=-1)

grid_model.fit(X, y)

# 4. Best score & parameters 
print('best score =', grid_model.best_score_) 
# best score = 0.94214794181368

print('best parameters =', grid_model.best_params_)
'''
best parameters = {'max_depth': None, 
                   'max_features': 'sqrt', 
                   'min_samples_leaf': 1, 
                   'min_samples_split': 2, 
                   'n_estimators': 150}
'''

# best model 생성 
best_model = RandomForestClassifier(max_depth=None,
                                    max_features='sqrt',
                                    min_samples_leaf=1,
                                    min_samples_split= 2,
                                    n_estimators=150)
best_model.fit(X = X_train, y = y_train)


# best model 평가 
best_model.score(X = X_test, y = y_test) # 0.96111111111


###################################################################################################

'''
- XGBoost 앙상블 모델 테스트
- Anaconda Prompt에서 패키지 설치 
  pip install xgboost
'''

from xgboost import XGBClassifier # model
from xgboost import plot_importance # 중요변수(x) 시각화  
from sklearn.datasets import make_blobs # 클러스터 생성 dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


dir(XGBClassifier)

# 1. 데이터셋 로드 : blobs
X, y = make_blobs(n_samples=2000, n_features=4, centers=2, 
                   cluster_std=2.5, random_state=123)
'''
n_samples : 표본크기 
n_features : x변수 개수 
centers : y변수 class 개수 
  centers=3 : 다항분류
  centers=2 : 이항분류 
cluster_std : 데이터 복잡도 지정 
'''

X.shape # (2000, 4)
y # array([1, 1, 0, ..., 0, 0, 2])

# blobs 데이터 분포 시각화 
plt.title("three cluster dataset")
plt.scatter(X[:, 0], X[:, 1], s=100, c=y,  marker='o') # color = y범주
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# 2. 훈련/검정 데이터셋 생성
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)



# 3. XGBOOST model : 100개 tree
model = XGBClassifier(objective='binary:logistic') # objective : 활성함수  
'''
이항분류기 : sigmoid 함수 
binary-class classification : objective='binary:logistic' 
다항분류기 : softmax 함수 
multi-class classification : objective='multi:softprob' 
'''

# train data 이용 model 생성  
eval_set = [(X_test, y_test)] # 평가셋 
model.fit(X_train, y_train, eval_set = eval_set, 
          eval_metric='error', verbose=True) 
'''
eval_metric : 학습과정에서 평가방법 
binary-class classification 평가방법 : eval_metric = 'error'
multi-class classification 평가방법 : eval_metric = 'merror'

eval_set : 평가셋 
eval_metric : 평가방법(훈련셋 -> 평가셋) 
'''


# 4. model 평가 
y_pred = model.predict(X_test) 
acc = accuracy_score(y_test, y_pred)
print('분류정확도 =', acc) 
# 분류정확도 = 0.925
# 분류정확도 = 0.9983333333333333

report = classification_report(y_test, y_pred)
print(report)


# 5. fscore 중요변수 시각화  
fscore = model.get_booster().get_fscore()
print("fscore:",fscore) 


# 중요변수 시각화
plot_importance(model) 
plt.show()

###################################################################################################

"""
 - XGBoost 회귀트리 예
"""
#from xgboost import XGBClassifier # 분류트리 모델 
from xgboost import XGBRegressor # 회귀트리 모델 
from xgboost import plot_importance # 중요변수 시각화 

import pandas as pd # dataset
from sklearn.model_selection import train_test_split # split 
from sklearn.metrics import mean_squared_error, r2_score # 평가 

# 스케일링 도구 
from sklearn.preprocessing import minmax_scale # 정규화(0~1) : X변수 
import numpy as np # 로그변환 + 난수 : y변수 



### 1. dataset load & preprocessing 

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
 14  CAT. MEDV  506 non-null    int64   : 제외  
'''

X = boston.iloc[:, :13] # 독립변수 
X.shape # (506, 13)

X.mean(axis = 0)

y = boston.MEDV # 종속변수 
y.shape #(506,)

# x,y변수 스케일링 
X_scaled = pd.DataFrame(minmax_scale(X), columns=X.columns) # 정규화
X_scaled.mean()

y = np.log1p(y) # 로그변환    


###  2. train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.3)


### 3. model 생성 : 활성함수 = SE
model = XGBRegressor().fit(X=X_train, y=y_train) 
# objective='reg:squarederror' : 활성함수
print(model)


### 4. 중요변수 확인 
fscore = model.get_booster().get_fscore()

# 중요변수 시각화  
plot_importance(model, max_num_features=13) # 13개까지 나타냄 
# 중요변수 6개 : 범죄율(CRIM), 방개수(RM), 하위계층(LSTAT), 연식(AGE), 흑인비율(B), 직업접근성(DIS)

# X:연속형(비율척도) vs y:연속형(비율척도)
import seaborn as sn 

boston.CRIM # X 
boston.MEDV # y

sn.scatterplot(data = boston, x=boston.CRIM, y = boston.MEDV)

# 비율척도 -> 명목척도(구간화)
boston.RM.min() # 3.561
boston.RM.max() # 8.78
'''
3 : 4미만 
4 : 4~5
5 : 5~6
6 : 6~7
7 : 7이상
'''

rm = boston.RM
rm_new = [] 

for r in rm :
    if r < 4 :
        rm_new.append(3)
    elif r > 4 and r < 5 :
        rm_new.append(4)
    elif r > 5 and r < 6 :
        rm_new.append(5)        
    elif r > 6 and r < 7 :
        rm_new.append(6)
    else :
       rm_new.append(7) 
       
rm[:5] # 비율척도       
rm_new[:5] # 구간화       

# 칼럼 추가 
boston['rm_new'] = rm_new

sn.barplot(data = boston, x=boston.rm_new, y = boston.MEDV)
# [해설] 대체적으로 방의 개수가 많을 수록 주택 가격이 높아진다.


### 5. model 평가 
y_pred = model.predict(X = X_val)  # 예측치 

# 평균제곱오차 : 0의 수렴 정도 
mse = mean_squared_error(y_val, y_pred)
print('MSE =',mse) 
# MSE = 0.020761046798285304

# 결정계수 : 1의 수렴 정도 
score = r2_score(y_val, y_pred)
print('r2 score =', score) 
# r2 score = 0.8228942487450536


### 6. model save & Testing 
import pickle # binary file 

# model file save 
pickle.dump(model, open('xgb_model.pkl', mode='wb'))

# model file load 
load_model = pickle.load(open('xgb_model.pkl', mode='rb'))


# final model Test 
idx = np.random.choice(a=len(X_scaled), size=200, replace=False) # test set 만들기 

X_test, y_test = X_scaled.iloc[idx], y[idx]

y_pred = load_model.predict(X = X_test) # new test set 

score = r2_score(y_test, y_pred)
print('r2 score =', score) 
# r2 score = 0.9564870041583043


###################################################################################################

"""
1. XGBoost Hyper parameters 
2. model 학습 조기 종료 
3. Best Hyper parameters
"""

from xgboost import XGBClassifier # model 
from sklearn.datasets import load_breast_cancer # 이항분류 dataset 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import accuracy_score, classification_report # 평가


### 1. XGBoost Hyper parameters 

# 1) dataset load 
cancer = load_breast_cancer()

x_names = cancer.feature_names
print(x_names, len(x_names)) # 30
y_labels = cancer.target_names
print(y_labels) # ['malignant' 'benign'] : 이항

X, y = load_breast_cancer(return_X_y=True)


# 2) train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)


# 3) default model 생성 
model = XGBClassifier(n_estimators=100).fit(X_train, y_train) # default 

print(model) # 주요 default parameters 
'''
1.colsample_bytree=1 : 트리 모델 생성 시 훈련셋 샘플링 비율(보통 : 0.6 ~ 1)
2.learning_rate=0.3 : 학습율(보통 : 0.01~0.1) = 0의 수렴속도 
3.max_depth=6 : 트리의 깊이(클 수록 성능이 좋아짐, 과적합 영향)
4.min_child_weight=1 : 자식 노드 분할을 결정하는 가중치(Weight)의 합
  - 값을 작게하면 더 많은 자식 노드 분할(과적합 영향)
5. n_estimators=100 결정 트리 개수(default=100), 많을 수록 고성능
6. objective='binary:logistic', 'multi:softprob'
과적합 조절 : max_depth 작게, min_child_weight 크게 
'''               

# 4) model 평가 
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(acc) # 0.9532163742690059



### 2. 학습 조기 종료 model 생성 
xgb = XGBClassifier(colsample_bytree=1,
                    learning_rate=0.3,
                    max_depth=6,
                    min_child_weight=1,
                    n_estimators=500) # 100 -> 500

eval_set = [(X_test, y_test)]  

# 1) 학습조기종료 model 생성 
es_model = xgb.fit(X=X_train, y=y_train, # 훈련셋 
                eval_set=eval_set, # 검증셋 
                eval_metric='error', # 평가기준 
                early_stopping_rounds=80, # 기본 tree 개수 
                verbose=True) # 콘솔 출력 

# [112]	validation_0-error:0.04094
'''
early_stopping_rounds=80 : 80tree 학습 후 오차(error) 변화 없으면
   특정 시점에서 조기종료  
'''
# 2) model 평가 
y_pred = es_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(acc) # 0.9590643274853801

report = classification_report(y_test, y_pred)
print(report)


### 3. Best Hyper parameters
from sklearn.model_selection import GridSearchCV # class 

# default parameters 
xgb = XGBClassifier()

params = {'colsample_bytree': [0.5, 0.7, 1],
          'learning_rate' : [0.01, 0.3, 0.5],
          'max_depth' : [5, 6, 7],
          'min_child_weight' : [1, 3, 5],
          'n_estimators' : [100, 200, 300]} # dict


gs = GridSearchCV(estimator = xgb, 
             param_grid = params, cv=5)

model = gs.fit(X=X_train, y=y_train, eval_metric='error',
       eval_set = eval_set, verbose=True)


print('best score =', model.best_score_)
# best score = 0.9724683544303797

print('best parameters :', model.best_params_)
'''
{'colsample_bytree': 0.5, 
 'learning_rate': 0.5, 'max_depth': 5, 
 'min_child_weight': 5, 'n_estimators': 100}
'''


###################################################################################################

"""
1. XGBoost Hyper Parameter
2. 학습조기종료(early_stopping)
3. Best Parameter
"""
from xgboost import XGBRegressor # model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd # dataset
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
 14  CAT. MEDV  506 non-null    int64   : 제외  
'''

X = boston.iloc[:, :13] # 독립변수 
X.shape # (506, 13)

y = boston.MEDV # 종속변수 
y.shape #(506,)

# x,y변수 스케일링 안됨 
X.mean() # 70.07396704469443
y.mean() # 22.532806324110677


# 스케일링 
X_scaled = pd.DataFrame(minmax_scale(X), columns=X.columns) # 정규화
y = np.log1p(y) # 로그변환    


#  2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=123)


#################################
## XGBoost Hyper Parameter
#################################
model = XGBRegressor(colsample_bytree=1,
                    learning_rate=0.3, 
                    max_depth=6, 
                    min_child_weight=1,
                    n_estimators=400) # objective='reg:squarederror'
'''
colsample_bytree=1 : 트리를 생성할때 훈련셋에서 feature 샘플링 비율(보통 :0.6~0.9)
learning_rate=0.1 : 학습율(보통 : 0.01~0.2)
max_depth=3 : tree 깊이, 과적합 영향
min_child_weight=1 : 최소한의 자식 노드 가중치 합(자식 노드 분할 결정), 과적합 영향
# - 트리 분할 단계에서 min_child_weight 보다 더 적은 노드가 생성되면 트리 분할 멈춤
n_estimators=100 : tree model 수 
objective='reg:squarederror'
'''

#################################
## 학습조기종료(early_stopping)
#################################
evals = [(X_test, y_test)]
model.fit(X_train, y_train, # 훈련셋 
          eval_metric='rmse', # 평가방법
                early_stopping_rounds=100, 
                eval_set=evals, # 평가셋 
                verbose=True)

# [151]	validation_0-rmse:0.16314

'''
early_stopping_rounds=100 : 조기종료 파라미터
 - 100개 tree model 학습과정에서 성능평가 지수가 향상되지 않으면 조기종료
'''

# 4. model 평가 
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('mse =', mse) 
# mse = 0.02653424888554659

score = r2_score(y_test, y_pred)
print('score=', score) 
# score= 0.8132951769205492

#################################
## Best Parameter
#################################
from sklearn.model_selection import GridSearchCV

# 기본 모델 객체 생성 : default parameter 
model = XGBRegressor(n_estimators=100,                  
                    objective='reg:squarederror')

params = {'max_depth':[3, 5, 7], 
          'min_child_weight':[1, 3],
          'n_estimators' : [100, 150, 200],
          'colsample_bytree':[0.5, 0.7], 
          'learning_rate':[0.01, 0.5, 0.1]}


# GridSearch model  
grid_model = GridSearchCV(estimator=model, param_grid=params, cv=5)

# GridSearch model 학습 : 훈련셋  
grid_model.fit(X=X_train, y= y_train)

print('best score =', grid_model.best_score_) 
# best score = 0.8728546505525833

print('best parameter =', grid_model.best_params_)
'''
{'colsample_bytree': 0.7, 
 'learning_rate': 0.1, 'max_depth': 3, 
 'min_child_weight': 3, 'n_estimators': 200}
'''













































































