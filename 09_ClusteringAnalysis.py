'''
계층적 군집분석(Hierarchical Clustering) 
 - 상향식(Bottom-up)으로 계층적 군집 형성 
 - 유클리드안 거리계산식 이용 
 - 숫자형 변수만 사용
'''

from scipy.cluster.hierarchy import linkage, dendrogram # 군집분석 도구 
import matplotlib.pyplot as plt # 시각화 
from sklearn.datasets import load_iris # dataset
import pandas as pd # DataFrame


# sample data : ppt.12
df = pd.DataFrame({'x' : [1, 2, 2, 4, 5],
                   'y' : [1, 1, 4, 3, 4]})

# 계층적 군집모형 
model = linkage(df, method='single')

print(model)
'''
  p         q          거리       노드수 
[[0.         1.         1.         2.        ]  
 [3.         4.         1.41421356 2.        ]
 [2.         6.         2.23606798 3.        ]
 [5.         7.         2.82842712 5.        ]]
'''

# 덴드로그램 시각화 
plt.figure(figsize = (25, 10))
dendrogram(model)
plt.show()



# 1. dataset loading
iris = load_iris() # Load the data

X = iris.data # x변수 
y = iris.target # y변수(target) - 숫자형 : 거리계산 

# X + y 결합  
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = y # target 추가 


# 2. 계층적 군집분석 
clusters = linkage(iris_df, method='single')
'''
군집화 방식 : ppt.10 ~ 11 참고 
method = 'single' : 단순연결(default)
method = 'complete' : 완전연결 
method = 'average' : 평균연결
method = 'centroid' : 두 중심점의 거리 
'''
print(clusters)
clusters.shape # (149, 4)


# 3. 덴드로그램(dendrogram) 시각화 : 군집수 사용자가 결정 
plt.figure(figsize = (25, 10))
dendrogram(clusters)
plt.show()


from scipy.cluster.hierarchy import fcluster # 군집 자르기 도구 
import numpy as np # 클러스터 빈도수 

# 4. 군집(cluster) 자르기 : ppt.17 ~ 18 참고
cut_cluster = fcluster(clusters, t=3, criterion='maxclust') 
cut_cluster # 1 ~ 3
len(cut_cluster) # 150

# 군집(cluster) 빈도수 
unique, counts = np.unique(cut_cluster, return_counts=True)
print(unique, counts)
# [1 2 3] [50 50 50]

# 5. DF 칼럼 추가 
iris_df['cluster'] = cut_cluster
iris_df


# 6. 계층적군집분석 시각화 
plt.scatter(iris_df['sepal length (cm)'], 
            iris_df['petal length (cm)'],
            c=iris_df['cluster'])
plt.show()


# 7. 각 군집별 특성분석 
group = iris_df.groupby('cluster')

group.size()
'''
cluster
1    50
2    50
3    50
'''

group.mean().T
'''
cluster                1      2      3
sepal length (cm)  5.006  5.936  6.588
sepal width (cm)   3.428  2.770  2.974
petal length (cm)  1.462  4.260  5.552
petal width (cm)   0.246  1.326  2.026
species            0.000  1.000  2.000

cluster1 : 꽃받침 길이/넓이, 꽃잎 길이/넓이 상대적으로 작음
cluster2 : 꽃받침 길이/넓이, 꽃잎 길이/넓이 중간 성향 
cluster3 : 꽃받침 길이/넓이, 꽃잎 길이/넓이 상대적으로 큼
'''

############################################################################################################

"""
kMeans 알고리즘 
 - 확인적 군집분석 
 - 군집수 k를 알고 있는 분석방법 
"""

import pandas as pd # DataFrame 
from sklearn.cluster import KMeans # model 
import matplotlib.pyplot as plt # 군집결과 시각화 
import numpy as np # array 


# 1. text file -> dataset 생성 
file = open('C:/ITWILL/4_Python_ML/data/testSet.txt')
lines = file.readlines() # list 반환 

print(lines)
len(lines) # 80

dataset = [] # 2차원(80x2)
for line in lines : # '1.658985\t4.285136\n'
    cols = line.split('\t') # tab키 기준 분리 
    
    rows = [] # 1줄 저장 : [1.658985, 4.285136] 
    for col in cols : # 칼럼 단위 추가 
        rows.append(float(col)) # 문자열 -> 실수형 변환  
        
    dataset.append(rows) # [[1.658985, 4.285136],...,[-4.905566, 2.911070]]
        
print(dataset) # 중첩 list   

# list -> numpy(array) 
dataset_arr = np.array(dataset)

dataset_arr.shape # (80, 2)
print(dataset_arr)


# 2. numpy -> DataFrame(column 지정)
data_df = pd.DataFrame(dataset_arr, columns=['x', 'y'])
data_df.info()
'''
RangeIndex: 80 entries, 0 to 79
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   x       80 non-null     float64
 1   y       80 non-null     float64
'''

plt.scatter(data_df['x'], data_df['y'])
plt.show()

# 3. KMeans model 생성 : k=4
obj = KMeans(n_clusters=4, max_iter=300, algorithm='auto')

model = obj.fit(data_df) # 학습 수행 


# 예측치 생성 
pred = model.predict(data_df) # test set 
print(pred) # 0 ~ 3
len(pred)

# 군집 중앙값 
centers = model.cluster_centers_
print(centers)
'''
        x           y
0 [[-3.38237045 -2.9473363 ]
1  [ 2.6265299   3.10868015]
2  [ 2.80293085 -2.7315146 ]
3  [-2.46154315  2.78737555]]
'''

# clusters 시각화 : 예측 결과 확인 
data_df['predict'] = pred # 칼럼추가 

data_df

# 산점도 
plt.scatter(x=data_df['x'], y=data_df['y'], 
            c=data_df['predict'])

# 중앙값 추가 
plt.scatter(x=centers[:,0], y=centers[:,1], 
            c='r', marker='D')
plt.show()


# 각 군집별 특성 분석 
group = data_df.groupby('predict')

group.size()
'''
0    20
1    20
2    20
3    20
'''

group.mean()
'''
                x         y
predict                    
0       -3.382370 -2.947336 : x,y값이 가장 작음 
1        2.626530  3.108680 : x,y값이 가장 큼   
2        2.802931 -2.731515 : x값이 가장 큼 
3       -2.461543  2.787376 : y값이 가장 큼 
'''

############################################################################################################

'''
UCI ML Repository 데이터셋 url
https://archive.ics.uci.edu/ml/datasets.php
'''

### 기본 라이브러리 불러오기
import pandas as pd
pd.set_option('display.max_columns', 100) # 콘솔에서 보여질 최대 칼럼 개수 
import matplotlib.pyplot as plt



### [Step 1] 데이터 준비 : 도매 고객 데이터셋 
'''
 - 도매 유통업체의 고객 관련 데이터셋으로 다양한 제품 범주에 대한 연간 지출액을 포함  
 - 출처: UCI ML Repository
'''
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(uci_path)
df.info() # 변수 및 자료형 확인
'''
RangeIndex: 440 entries, 0 to 439
Data columns (total 8 columns):
 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   Channel           440 non-null    int64 : 유통업체 : Horeca(호텔/레스토랑/카페) 또는 소매(명목)
 1   Region            440 non-null    int64 : 지역 : Lisnon,Porto 또는 기타(명목) - 리스본,포르토(포르투갈)  
 2   Fresh             440 non-null    int64 : 신선함 : 신선 제품에 대한 연간 지출(연속)
 3   Milk              440 non-null    int64 : 우유 : 유제품에 대한 연간 지출(연속)
 4   Grocery           440 non-null    int64 : 식료품 : 식료품에 대한 연간 지출(연속)
 5   Frozen            440 non-null    int64 : 냉동 제품 : 냉동 제품에 대한 연간 지출(연속)
 6   Detergents_Paper  440 non-null    int64 : 세제-종이 : 세제 및 종이 제품에 대한 연간 지출(연속)
 7   Delicassen        440 non-null    int64 : 델리카슨 : 델리카트슨(수입식품) 제품(연속)
'''


### [Step 2] 데이터 탐색

# 데이터 살펴보기
print(df.head())  

# 명목형 변수 
df.Channel.value_counts() # 유통업체
'''
1    298 : Horeca
2    142 : 소매 
'''

df.Region.value_counts() # 유통 지역
'''
3    316 : 기타 
1     77 : Lisnon
2     47 : Porto 
'''

# 연속형 변수 
df.describe() # 나머지 연속형 변수(각 변수 척도 다름) 


### [Step 3] 데이터 전처리

# 분석에 사용할 변수 선택
X = df.copy()

# 설명변수 데이터 정규화
from sklearn.preprocessing import StandardScaler # 표준화 
X = StandardScaler().fit_transform(X)


### [Step 4] k-means 군집 모형 - sklearn 사용

# sklearn 라이브러리에서 cluster 군집 모형 가져오기
from sklearn.cluster import KMeans

# 모형 객체 생성 : k=5 
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300, 
                random_state=45) # cluster 5개 
'''
Parameters
----------
n_clusters : int, default=8
n_init : int, default=10 - centroid seeds
max_iter : int, default=300
random_state : model seed value 
'''
        
# 모형 학습
kmeans.fit(X)  # KMeans(n_clusters=5) 



# 군집 예측 
cluster_labels = kmeans.labels_ # 예측된 레이블(Cluster 번호)    
print(cluster_labels) # 0 ~ 4



# 데이터프레임에 예측된 레이블 추가
df['Cluster'] = cluster_labels
print(df.head())   

# 상관관계 분석 
r = df.corr() # 상관계수가 높은 변수 확인 
r['Grocery']
 
# 그래프로 표현 - 시각화
df.plot(kind='scatter', x='Grocery', y='Detergents_Paper', c='Cluster', 
        cmap='Set1', colorbar=True, figsize=(15, 10))
plt.show()  

# 각 클러스터 빈도수 : 빈도수가 적은 클러스터 제거 
print(df.Cluster.value_counts())
'''
2    211
0    126
1     92
4     11 -> 제거 
3      2 -> 제거 
'''

# 새로운 dataset 만들기 : 3,4번 클러스터 제거 예 
new_df = df[~((df['Cluster'] == 3) | (df['Cluster'] == 4))]

new_df.shape # (427, 9)


# 새로운 dataset 만들기 : 산점도 
new_df.plot(kind='scatter', x='Grocery', y='Detergents_Paper', c='Cluster', 
        cmap='Set1', colorbar=True, figsize=(15, 10))
plt.show() 


### [Step 5] 각 cluster별 특성 분석
group = new_df.groupby('Cluster')

group.size()
'''
0    125 -> cluster1
1    211 -> cluster2
2     91 -> cluster3
'''

# 군집별 subset 만들기 
cluster1 = new_df[new_df.Cluster == 0]
cluster2 = new_df[new_df.Cluster == 1]
cluster3 = new_df[new_df.Cluster == 2]

# cluster1 : 명목척도 
cluster1.Channel.value_counts() # 유통업체 : 소매업
cluster1.Region.value_counts() # 지역 : 전지역       

# cluster2 : 명목척도 
cluster2.Channel.value_counts() # 유통업체 : Horeca 
cluster2.Region.value_counts() # 지역 : 기타

# cluster3 : 명목척도 
cluster3.Channel.value_counts() # 유통업체 : Horeca
cluster3.Region.value_counts() # 지역 : 리스본, 포르토

# 연속형 변수(상관계수 기준) : Grocery, Detergents_Paper, Milk
cols = ['Grocery', 'Detergents_Paper', 'Milk']
cluster1[cols].mean()
# 14212.624, 6149.592, 8913.512 

cluster2[cols].mean()
# 3814.625592, 788.028436, 3297.379147 

cluster3[cols].mean()
# 4130.923077, 860.263736, 3254.714286
'''
cluster1 : 유통업체는 소매업, 지역은 전지역, 연간 지출액은 가장 많은 그룹 
cluster2 : 유통업체는 Horeca, 지역은 기타, 연간 지출액은 가장 적은 그룹
cluster3 : 유통업체는 Horeca, 지역은 리스본, 포르토, 연간 지출액은 중간 그룹
'''


############################################################################################################

'''
Best Cluster 찾는 방법 
'''

from sklearn.cluster import KMeans # model 
from sklearn.datasets import load_iris # dataset 
import matplotlib.pyplot as plt # 시각화 

# 1. dataset load 
X, y = load_iris(return_X_y=True)
print(X.shape) # (150, 4)
print(X)


# 2. Best Cluster 
size = range(1, 11) # k값 범위(1 ~ 10)
inertia = [] # 응집도 

for k in size : 
    obj = KMeans(n_clusters = k) 
    model = obj.fit(X)
    inertia.append(model.inertia_) 

'''
inertia value : 중심점와 각 포인트 간의 거리제곱합  
inertia value 작을 수록 응집도가 좋다.
'''
print(inertia)
'''
[681.3706, 152.3479517603579, 
 78.851441426146, 57.256009315718146, 
 46.44618205128206, 39.03998724608725, 
 34.75674963924964, 29.99042640692641, 
 28.042265254353495, 25.988974783479932]
'''

# 3. best cluster 찾기 
plt.plot(size, inertia, '-o')
plt.xticks(size)
plt.show()


