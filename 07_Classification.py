"""
 - 알려진 범주로 알려지지 않은 범주 분류 
 - 유클리드 거리계신식 이용 
"""

from sklearn.neighbors import KNeighborsClassifier # class

# 1.dataset 생성 : ppt.7 참고 
grape = [8, 5]   # 포도[단맛,아삭거림] - 과일(0)
fish = [2, 3]    # 생성[단맛,아삭거림] - 단백질(1)
carrot = [7, 10] # 당근[단맛,아삭거림] - 채소(2)
orange = [7, 3]  # 오랜지[단맛,아삭거림] - 과일(0)
celery = [3, 8]  # 셀러리[단맛,아삭거림] - 채소(2)
cheese = [1, 1]  # 치즈[단맛,아삭거림] - 단백질(1)

# x변수 : 알려진 그룹  
know = [grape,fish,carrot,orange,celery,cheese]  # 중첩 list

# y변수 : 알려진 그룹의 클래스
y_class = [0, 1, 2, 0, 2, 1] 

# 알려진 그룹의 클래스 이름(class name) 
class_label = ['과일', '단백질', '채소'] 
 

# 2. 분류기 
knn = KNeighborsClassifier(n_neighbors = 3) # k=3 
model = knn.fit(X = know, y = y_class) 
know # [[8, 5], [2, 3], [7, 10], [7, 3], [3, 8], [1, 1]]

# 3. 분류기 평가 
x1 = 4 # 단맛(1~10) : 4 -> 8 -> 2
X2 = 8 # 아삭거림(1~10) : 8 -> 2 -> 3

# 분류대상 
unKnow = [[x1, X2]]  # 중첩 list : [[4, 8]]

# class 예측 
y_pred = model.predict(X = unKnow)
print(y_pred) # [2] -> [0] -> [1]

print('분류결과 : ',class_label[y_pred[0]]) # 분류결과 : 채소
print('분류결과 : ',class_label[y_pred[0]]) # 분류결과 : 과일
print('분류결과 : ',class_label[y_pred[0]]) # 분류결과 : 단백질



# know(6) vs unKnow(1) 

import numpy as np 

# 1. 다차원 배열 변환 
know_arr = np.array(know)
unKnow_arr = np.array(unKnow)

know_arr.shape # (6, 2)
unKnow_arr.shape # (1, 2)

# 유클리드 거리계산식 : 차(-) -> 제곱 -> 합 -> 제곱근 
diff = know_arr - unKnow_arr # 1) 차(-)
diff

diff_square = np.square(diff) # 2) 제곱
diff_square

diff_square_sum = diff_square.sum(axis = 1) # 3) 합
diff_square_sum # [25, 29, 13, 34,  1, 58]

distance = np.sqrt(diff_square_sum) # 4) 제곱근 

print(distance) # k=3
# [5.  5.38516481 3.60555128 5.83095189 1. 7.61577311]


# 오름차순 색인정렬 : k=3
idx = distance.argsort()[:3] # [4, 2, 0, 1, 3, 5]
idx # [4, 2, 0]

y_class = [0, 1, 2, 0, 2, 1] 

for i in idx :
    #print(y_class[i])
    print(class_label[y_class[i]])
'''
채소
채소
과일
'''

##########################################################################################################################

'''
Naive Bayes 이론에 근거한 통계적 분류기

 1. GaussianNB  : x변수가 연속형이고, 정규분포인 경우 
 2. MultinomialNB : x변수가 단어 빈도수(텍스트 데이터)를 분류할 때 적합
'''

###############################
### news groups 분류 
###############################

#from sklearn.naive_bayes import GaussianNB # x변수가 연속형  
from sklearn.naive_bayes import MultinomialNB # tfidf 문서분류
from sklearn.datasets import fetch_20newsgroups # news 데이터셋 
from sklearn.feature_extraction.text import TfidfVectorizer# dtm(희소행렬) 
from sklearn.metrics import accuracy_score, confusion_matrix # model 평가 


# 1. dataset 가져오기 
newsgroups = fetch_20newsgroups(subset='all') # train/test load 

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
print(newsgroups.target_names) # 20개 뉴스 그룹 
print(len(newsgroups.target_names)) # 20개 뉴스 그룹 


# 2. train set 선택 : 4개 뉴스 그룹  
#cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
cats = newsgroups.target_names[:4]

news_train = fetch_20newsgroups(subset='train',categories=cats)
news_data = news_train.data # texts
news_target = news_train.target # 0 ~ 3


# 3. DTM(sparse matrix)
obj = TfidfVectorizer()
sparse_train = obj.fit_transform(news_data)
'''
<2245x62227 sparse matrix of type '<class 'numpy.float64'>'
	with 339686 stored elements in Compressed Sparse Row format>
'''    
sparse_train.shape # (2245, 62227)


# 4. NB 모델 생성 
nb = MultinomialNB() # alpha=.01 (default=1.0)
model = nb.fit(sparse_train, news_target) # 훈련셋 적용 


# 5. test dataset 4개 뉴스그룹 대상 : 희소행렬
news_test = fetch_20newsgroups(subset='test', categories=cats)
news_test_data = news_test.data # text 
len(news_test_data) # 1,494

y_true = news_test.target


sparse_test = obj.transform(news_test_data) # 함수명 주의  
sparse_test.shape # (1494, 62227)


# 6. model 평가 
y_pred = model.predict(sparse_test) # 예측치 

acc = accuracy_score(y_true, y_pred)
print('accuracy =', acc) # accuracy = 0.852074

# 2) confusion matrix
con_mat = confusion_matrix(y_true, y_pred)
print(con_mat)
'''
    0     1   2   3
0 [[312   2   1   4]
1  [ 12 319  22  36]
2  [ 16  26 277  75]
3  [  1   8  18 365]]
'''

312 / (312 +  2  +  1 +  4) # 0.9780


##########################################################################################################################

"""
 - 선형 SVM, 비선형 SVM
"""

from sklearn.svm import SVC # svm model 
from sklearn.datasets import load_breast_cancer # dataset 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import accuracy_score # 평가 

# 1. dataset load 
X, y = load_breast_cancer(return_X_y= True)
X.shape # (569, 30)

X.var() # 52119.70516752481

y # 0 or 1 

# 2. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. 비선형 SVM 모델 
svc = SVC(C=1.0, kernel='rbf', gamma='scale')
'''
기본 parameter
 C=1.0 : cost(오분류) 조절 : 결정경계 위치 조정
 kernel='rbf' : 커널트릭 함수 
  -> kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
 gamma='scale' : 결정경계 모양 조절 조정 
  -> {'scale', 'auto'} or float
  -> gamma='scale' : 1 / (n_features * X.var())
  -> gamma='auto' : 1 / n_features
'''

n_features = 30 # 독립변수 개수  
gamma_scale = 1 / (n_features * X.var()) # 0.00000063955337
gamma_auto = 1 / n_features # 0.03333333333333333


model = svc.fit(X=X_train, y=y_train)


# model 평가 
y_pred = model.predict(X = X_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy =',acc) # accuracy = 0.9005847953216374


# 4. 선형 SVM  
obj2 = SVC(C=1.0, kernel='linear') # gamma 사용 안함

model2 = obj2.fit(X=X_train, y=y_train)

# model 평가 
y_pred = model2.predict(X = X_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy =',acc) # accuracy = 0.9707602339181286

##########################################################################################################################

#########################################
### 선형 vs 비선형 커널 함수 
#########################################

'''
'linear' 커널
선형 커널은 데이터를 선형으로 분리하려고 시도한다. 즉, 데이터를 직선으로 나누는 것을 목표로 한다.
이 커널은 클래스 간의 경계가 선형일 때 유용하다.
'linear' 커널은 비교적 간단하며, 매개변수 튜닝이 적은 편이다.

'rbf' (Radial Basis Function) 커널
'rbf' 커널은 비선형 데이터를 다루는 데 유용하다. 비선형 데이터는 선형 경계로는 분리하기 어려울 때 사용된다.
이 커널은 데이터를 고차원 특징 공간으로 매핑하여 선형 분리 가능한 상태로 변환한다.
'rbf' 커널은 SVM에서 가장 일반적으로 사용되는 커널 중 하나이며, C와 gamma에 민감하며, 
매개변수 튜닝이 중요하다. C는 마진 오류에 대한 패널티를 조절하고, gamma는 커널의 모양을 제어한다.
'''

from sklearn.svm import SVC # svm model 
from sklearn.datasets import make_circles # 중첩된 원형 dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. 데이터셋 로드 : 원형 데이터셋 
X, y = make_circles(noise=0.05, n_samples=200, random_state=123)

X.shape # (200, 2)
y.shape # (200,)
y #  array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0,...])

plt.scatter(X[:, 0], X[:, 1], s=100, c=y,  marker='o') # color = y범주
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# 2. 훈련/검정 데이터셋 생성
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)


# 3. 선형 모델 적용 
obj = SVC(C=1.0, kernel='linear')

model = obj.fit(X=X_train, y=y_train)
model.score(X=X_train, y=y_train) # 0.5642857142857143
model.score(X=X_test, y=y_test) # 0.35


# 비선형 모델 적용 
obj2 = SVC(C=1.0, kernel='rbf')

model2 = obj2.fit(X=X_train, y=y_train)

model2.score(X=X_train, y=y_train) # 0.9714285714285714
model2.score(X=X_test, y=y_test) # 0.9833333333333333


##########################################################################################################################

"""
 - Grid Search : best parameger 찾기 
"""

from sklearn.svm import SVC # svm model 
from sklearn.datasets import load_breast_cancer # dataset 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import accuracy_score # 평가 

# 1. dataset load 
X, y = load_breast_cancer(return_X_y= True)
X.shape # (569, 30)
y # 0 or 1 

# 2. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. 비선형 SVM 모델 
svc = SVC(C=1.0, kernel='rbf', gamma='scale') # 기본 모델 
'''
기본 parameter
 C=1.0 : cost(오분류) 조절 : 결정경계 위치 조정
 kernel='rbf' : 커널트릭 함수 
  -> kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
 gamma='scale' : 결정경계 모양 조절 조정 
  -> {'scale', 'auto'} or float
  -> gamma='scale' : 1 / (n_features * X.var())
  -> gamma='auto' : 1 / n_features
'''

model = svc.fit(X=X_train, y=y_train) # 기본 모델 


# model 평가 
y_pred = model.predict(X = X_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy =',acc) # accuracy = 0.9005847953216374


# 4. 선형 SVM : 선형분류 
obj2 = SVC(C=1.0, kernel='linear')

model2 = obj2.fit(X=X_train, y=y_train)

# model 평가 
y_pred = model2.predict(X = X_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy =',acc) # accuracy = 0.9707602339181286


###############################
### Grid Search 
###############################

from sklearn.model_selection import GridSearchCV 

parmas = {'kernel' : ['rbf', 'linear'],
          'C' : [0.01, 0.1, 1.0, 10.0, 100.0],
          'gamma': ['scale', 'auto']} # dict 정의 

# 5. GridSearch model   
grid_model = GridSearchCV(model, param_grid=parmas, 
                   scoring='accuracy',cv=5, n_jobs=-1).fit(X, y)
'''
param_grid : 찾을 파라미터 
scoring='accuracy' : model 평가 방법 
cv=5 : 5겹 교차검정 
n_jobs=-1  : cpu 사용수 
'''

dir(grid_model)

# 1) Best score 
print('best score =', grid_model.best_score_)
# best score = 0.9631268436578171

# 2) Best parameters 
print('best parameters =', grid_model.best_params_)
# best parameters = {'C': 100.0, 'gamma': 'scale', 'kernel': 'linear'}

# 3) Best model 
svc = SVC(C=100, kernel='linear', gamma='scale')

best_model = svc.fit(X=X_train, y=y_train)

y_pred = best_model.predict(X = X_test)

acc = accuracy_score(y_test, y_pred)
print(acc) # 0.9766081871345029
