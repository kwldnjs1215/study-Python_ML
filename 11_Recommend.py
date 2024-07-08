'''
영화 추천 시스템 알고리즘
 - 추천 대상자 : Toby   
 - 유사도 평점 = 미관람 영화평점 * Toby와의 유사도
 - 추천 영화 예측 = 유사도 평점 / Toby와의 유사도
'''

import pandas as pd

# 데이터 가져오기 
ratings = pd.read_csv(r'C:\ITWILL\4_Python_ML\data\movie_rating.csv')
print(ratings) 


### 1. pivot table 작성 : row(영화제목), column(평가자), cell(평점)
movie_ratings = pd.pivot_table(ratings,
               index = 'title', # 행
               columns = 'critic', # 열 
               values = 'rating').reset_index()

print(movie_ratings)  
'''
critic      title  Claudia  Gene  Jack  Lisa  Mick  Toby
0         Just My      3.0   1.5   NaN   3.0   2.0   NaN
1            Lady      NaN   3.0   3.0   2.5   3.0   NaN
2          Snakes      3.5   3.5   4.0   3.5   4.0   4.5
3        Superman      4.0   5.0   5.0   3.5   3.0   4.0
4       The Night      4.5   3.0   3.0   3.0   3.0   NaN
5          You Me      2.5   3.5   3.5   2.5   2.0   1.0
'''

        
### 2. 사용자 유사도 계산(상관계수 R)  
sim_users = movie_ratings.iloc[:,1:].corr().reset_index() # corr(method='pearson')
print(sim_users) 
'''
critic   critic   Claudia      Gene      Jack      Lisa      Mick      Toby
0       Claudia  1.000000  0.314970  0.028571  0.566947  0.566947  0.893405
1          Gene  0.314970  1.000000  0.963796  0.396059  0.411765  0.381246
2          Jack  0.028571  0.963796  1.000000  0.747018  0.211289  0.662849
3          Lisa  0.566947  0.396059  0.747018  1.000000  0.594089  0.991241
4          Mick  0.566947  0.411765  0.211289  0.594089  1.000000  0.924473
5          Toby  0.893405  0.381246  0.662849  0.991241  0.924473  1.000000
'''

sim_users['Toby'] # Toby vs other 
'''
0    0.893405
1    0.381246
2    0.662849
3    0.991241 -> Lisa
4    0.924473 -> Mick
'''

test = ratings[ratings['critic'].isin(['Lisa','Mick'])]
test

### 3. Toby 미관람 영화 추출  
# 1) movie_ratings table에서 title, Toby 칼럼으로 subset 작성 
toby_rating = movie_ratings[['title', 'Toby']]  
print(toby_rating)
'''
critic title     Toby
0    Just My     NaN
1       Lady     NaN
2     Snakes     4.5
3   Superman     4.0
4  The Night     NaN
5     You Me     1.0
'''

# 2) Toby 미관람 영화제목 추출 
# 형식) DF.칼럼[DF.칼럼.isnull()]
toby_not_see = toby_rating.title[toby_rating.Toby.isnull()] 
print(toby_not_see) # rating null 조건으로 title 추출 
'''
0      Just My
1         Lady
4    The Night
'''
type(toby_not_see)


# 3) raw data에서 Toby 미관람 영화만 subset 생성 
rating_t = ratings[ratings.title.isin(toby_not_see)] # 3편 영화제목 
print(rating_t)
'''
     critic      title  rating
0      Jack       Lady     3.0
4      Jack  The Night     3.0
5      Mick       Lady     3.0
:
30     Gene  The Night     3.0
'''


# 4. Toby 미관람 영화 + Toby 유사도 join
# 1) Toby 유사도 추출 
toby_sim = sim_users[['critic','Toby']] # critic vs Toby 유사도 
toby_sim
'''
critic   critic      Toby
0       Claudia  0.893405
1          Gene  0.381246
2          Jack  0.662849
3          Lisa  0.991241
4          Mick  0.924473
5          Toby  1.000000
'''

# 2) 평가자 기준 병합  
rating_t = pd.merge(rating_t, toby_sim, on='critic')
print(rating_t)
'''
     critic      title  rating      Toby
0      Jack       Lady     3.0  0.662849
1      Jack  The Night     3.0  0.662849
2      Mick       Lady     3.0  0.924473
'''


### 5. 유사도 평점 계산 = Toby미관람 영화 평점 * Tody 유사도 
rating_t['sim_rating'] = rating_t.rating * rating_t.Toby
print(rating_t)
'''
     critic      title  rating      Toby  sim_rating
0      Jack       Lady     3.0    0.662849    1.988547
1      Jack  The Night     3.0    0.662849    1.988547
2      Mick       Lady     3.0    0.924473    2.773420
[해설] Toby 미관람 영화평점과 Tody유사도가 클 수록 유사도 평점은 커진다.
'''

### 6. 영화제목별 rating, Toby유사도, 유사도평점의 합계
group_sum = rating_t.groupby(['title']).sum() # 영화 제목별 합계
'''
           rating      Toby  sim_rating
title                                  
Just My       9.5  3.190366    8.074754
Lady         11.5  2.959810    8.383808
The Night    16.5  3.853215   12.899752
'''
 
### 7. Toby 영화추천 예측 = 유사도평점합계 / Tody유사도합계
print('\n*** 영화 추천 결과 ***')
group_sum['predict'] = group_sum.sim_rating / group_sum.Toby
print(group_sum)
'''
           rating      Toby  sim_rating   predict
title                                            
Just My       9.5  3.190366    8.074754  2.530981
Lady         11.5  2.959810    8.383808  2.832550
The Night    16.5  3.853215   12.899752  3.347790 -> 추천영화 
'''

####################################################################################################

"""
SVD(Singular Value Decomposition) : 특이값을 이용한 행렬분해  
"""

import numpy as np 

# 1. A행렬 만들기 
A = np.arange(1, 7).reshape(2,3)
print(A)
'''
[[1 2 3]
 [4 5 6]]
'''
A.shape # (2, 3) : user(2) vs item(3)


# 2. svd : 행렬분해 
svd = np.linalg.svd(A) 
svd
'''
SVDResult(U=array([[-0.3863177 ,  0.92236578],
                  [-0.92236578, -0.3863177 ]]), : u행 
         S=array([9.508032  , 0.77286964]), : 특이값 
         Vh=array([[-0.42866713, -0.56630692, -0.7039467 ],
                   [-0.80596391, -0.11238241,  0.58119908],
                   [ 0.40824829, -0.81649658,  0.40824829]])) : v행                 '''
                                                                                  
u = svd[0] # u행렬
u.shape # (2, 2)  
d = svd[1] # 특이값 벡터  
d.shape # (2,)
v = svd[2] # v행렬  
v.shape # (3, 3) 

# @ : 행렬곱 
X = u @ np.diag(d) @ v[:2]  # (2,2) @ (2,2) = (2,2) @ (2,3)=(2,3)
X
'''
array([[1., 2., 3.],
       [4., 5., 6.]])
'''

####################################################################################################

"""
- 특이값 분해(SVD) 알고리즘 이용 추천 시스템

<준비물>
 scikit-surprise 패키지 설치 
> conda install -c conda-forge scikit-surprise
"""

import pandas as pd # csv file 
from surprise import SVD # SVD model 
from surprise import Reader, Dataset # SVD dataset 


# 1. dataset loading 
ratings = pd.read_csv('C:/ITWILL/4_Python_ML/data/movie_rating.csv')
print(ratings) #  평가자[critic]   영화[title]  평점[rating]


# 2. pivot table 작성 : row(영화제목), column(평가자), cell(평점)
print('movie_ratings')
movie_ratings = pd.pivot_table(ratings,
               index = 'title',
               columns = 'critic',
               values = 'rating').reset_index()


# 3. SVD dataset 
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings, reader)


# 4. train/test set 생성 
trainset = data.build_full_trainset() # 훈련셋 
testset = trainset.build_testset() # 평가셋  


# 5. SVD model 생성 
model = SVD(random_state=123).fit(trainset) # seed값 적용 
dir(model)
'''
predict() : 대상자 기준 예측 
test() : 전체 평가셋 예측 
'''

# 6. 전체 평가셋 예측 
all_pred = model.test(testset)
print(all_pred)
'''
(uid='사용자', iid='아이템', r_ui=실제평점, est=예측평점)
(uid='Jack', iid='Lady', r_ui=3.0, est=3.270719540168945, details={'was_impossible': False})
'''

# 7. Toby 사용자 영화 추천 예측 
user_id  = 'Toby'  # 대상자 
items = ['Just My','Lady','The Night'] # 추천 아이템   
actual_rating = 0 # 실제 평점 

for item_id in items :
    # model.predict('대상자', '아이템', 실제평점)
    svd_pred = model.predict(user_id, item_id, actual_rating)
    print(svd_pred)

'''
user: Toby       item: Just My    r_ui = 0.00   est = 2.88   {'was_impossible': False}
user: Toby       item: Lady       r_ui = 0.00   est = 3.27   {'was_impossible': False}
user: Toby       item: The Night  r_ui = 0.00   est = 3.30   {'was_impossible': False}
'''

####################################################################################################

'''
surprise Dataset 이용 
'''
import pandas as pd # DataFrame 생성 
from surprise import SVD, accuracy # SVD model 생성, 평가  
from surprise import Reader, Dataset # SVD dataset 생성  

############################
## suprise Dataset
############################

# 1. dataset loading 
ratings = pd.read_csv('C:/ITWILL/4_Python_ML/data/u.data', 
                      sep='\t', header=None)
print(ratings) 

# 칼럼명 수정 
ratings.columns = ['userId','movieId','rating','timestamp']
ratings.info() 
'''
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 4 columns):
 #   Column     Non-Null Count   Dtype
---  ------     --------------   -----
 0   userId     100000 non-null  int64
 1   movieId    100000 non-null  int64
 2   rating     100000 non-null  int64
 3   timestamp  100000 non-null  int64
'''

ratings = ratings.drop('timestamp', axis = 1)
 
# 2. pivot table 작성 : row(영화제목), column(평가자), cell(평점)
movie_ratings = pd.pivot_table(ratings,
               index = 'userId', # 사용자 
               columns = 'movieId', # 아이템 
               values = 'rating').reset_index()

movie_ratings.shape # (943, 1683)


# 3. SVD dataset 
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings, reader)


# 4. train/test split
from surprise.model_selection import train_test_split

# Dataset 자료이용 : 80 vs 20 
trainset, testset = train_test_split(data, random_state=0)


# 5. svd model
svd_model= SVD(random_state=123).fit(trainset) # 80000


# 5. 전체 testset 평점 예측
preds = svd_model.test(testset)
print(len(preds)) # 20,000

# 예측결과 출력 
print('user\tmovie\trating\test_rating')
for p in preds[:5] : 
    print(p.uid, p.iid, p.r_ui, p.est, sep='\t\t')
  
       
# 6. model 평가 
accuracy.mse(preds) 
accuracy.rmse(preds) 


# 7.추천대상자 평점 예측 

# 1) 추천대상자 선정 
movie_ratings.iloc[:5,:10] # 5행 10열 
'''
movieId   1    2    3    4    5    6    7    8    9    10
userId                                                   
1        5.0  3.0  4.0  3.0  3.0  5.0  4.0  1.0  5.0  3.0
2        4.0  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  2.0
3        NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
4        NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
5        4.0  3.0  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN <- 추천대상자 
'''

uid = '5' # 추천대상자 
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
items = movie_ratings.columns[1:11] # 추천대상 영화제목(10개) 
actual_rating = movie_ratings.iloc[4, 1:11].fillna(0) # 실제 평점 

for iid, r_ui in zip(items, actual_rating):
    pred = svd_model.predict(uid, iid, r_ui)
    print(pred)
    
'''
user: 5          item: 1          r_ui = 4.00   est = 3.96   {'was_impossible': False}
user: 5          item: 2          r_ui = 3.00   est = 3.37   {'was_impossible': False}
user: 5          item: 3          r_ui = 0.00   est = 3.23   {'was_impossible': False}
user: 5          item: 4          r_ui = 0.00   est = 3.63   {'was_impossible': False}
user: 5          item: 5          r_ui = 0.00   est = 3.38   {'was_impossible': False}
user: 5          item: 6          r_ui = 0.00   est = 3.83   {'was_impossible': False}
user: 5          item: 7          r_ui = 0.00   est = 3.83   {'was_impossible': False}
user: 5          item: 8          r_ui = 0.00   est = 4.04   {'was_impossible': False}
user: 5          item: 9          r_ui = 0.00   est = 4.02   {'was_impossible': False}
user: 5          item: 10         r_ui = 0.00   est = 3.93   {'was_impossible': False}
'''

    
