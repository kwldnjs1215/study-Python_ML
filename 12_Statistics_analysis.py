"""
statistics 모듈의 주요 함수 
  - 기술통계 : 대푯값, 산포도, 왜도/첨도 등   
"""

import statistics as st # 기술통계 
import pandas as pd # csv file 


# 기술통계 
path = r'C:\ITWILL\4_Python_ML\data'
dataset = pd.read_csv(path + '/descriptive.csv')
print(dataset.info())

x = dataset['cost'] # 구매비용 선택 

# 1. 대푯값
print('평균 =', st.mean(x)) 
print('중위수=', st.median(x)) 
print('낮은 중위수 = ', st.median_low(x))
print('높은 중위수 = ', st.median_high(x))
print('최빈수 =',  st.mode(x)) 


# 2. 산포도   
var = st.variance(x)
print('표본의 분산 = ', var) 
print('모집단의 분산 =', st.pvariance(x)) 

std = st.stdev(x)
print('표본의 표준편차 =', std) 
print('모집단의 표준편차 =', st.pstdev(x))

'''
표본표준편차(S) = 분산의 제곱근 
표본분산(S^2) = 표준편차의 제곱 
'''

# 사분위수 
print('사분위수 :', st.quantiles(x)) 
# 사분위수 : [4.425000000000001, 5.4, 6.2]



import scipy.stats as sts

# 3. 왜도/첨도 

# 1) 왜도 
sts.skew(x) # -0.1531779106237012 -> 오른쪽으로 기울어짐 
'''
왜도 = 0 : 좌우대칭 
왜도 > 0 : 왼쪽 치우침
왜도 < 0 : 오른쪽 치우침
'''

# 첨도 
sts.kurtosis(x) # -0.1830774864331568 = fisher 기준 
sts.kurtosis(x, fisher=False) # 2.816922513566843 = Pearson 기준 
'''
<fisher 기준>
첨도 = 0 : 정규분포 첨도 
첨도 > 0 : 뾰족함
첨도 < 0 : 완만함 

<Pearson 기준>
첨도 = 3 : 정규분포 첨도
첨도 > 3 : 뾰족함
첨도 < 3 : 완만함 
'''

# 히스토그램 + 밀도곡선
import seaborn as sn 

sn.displot(x, kind='hist', kde=True)


####################################################################################################

"""
step02_continuous_distribution.py

확률분포와 검정(test)

1. 연속확률분포와 정규성 검정 
  - 연속확률분포 : 정규분포, 균등분포, 카이제곱, T/Z/F 분포 등 
  - 정규분포의 검정   
2. 이산확률분포와 이항 검정 
  - 이산확률분포 : 베르누이분포, 이항분포, 포아송분포 등  
  - 이항분포의 검정 
"""

from scipy import stats # 확률분포 + 검정
import numpy as np 
import matplotlib.pyplot as plt # 확률분포의 시각화 


################################
### 정규분포와 검정(정규성 검정)
################################


# 단계1. 모집단에서 표본추출  

# 평균 및 표준 편차 설정
mu = 0 # 모평균 
sigma = 1 # 모표준편차 

# 표준정규분포에서 표본추출 : 연속확률변수 X 
X = stats.norm.rvs(loc=mu, scale=sigma, size=1000)  
'''
rvs(random variable sampling) : N개 표본추출 
loc : 모평균 
scale : 모표준편차
size : 표본크기 
'''
print(X)
X.min() # -2.645374591255763
X.max() # 3.1141056407151404


# 단계2. 확률밀도함수(pdf)    

# 밀도곡선을 위한 벡터 자료   
line = np.linspace(min(X), max(X), 100)

# 히스토그램 : 단위(밀도)
plt.hist(X, bins='auto', density=True) # 히스토그램 
plt.plot(line, stats.norm.pdf(line, mu, sigma), color='red') # 밀도곡선
plt.show()


# 단계3. 정규성 검정 : 가설검정 
''' 
 귀무가설(H0) : 정규분포와 차이가 없다.(부정적)
 대립가설(H1) : 정규분포와 차이가 있다.(긍정적)
'''

print(stats.shapiro(X))

statistic, pvalue = stats.shapiro(X)
print('검정통계량 = ', statistic) # 검정통계량 =  0.9980314119302677
print('유의확률 =', pvalue) # 유의확률 = 0.296036274956921

alpha = 0.05 # 유의수준(알파) : 5% 
'''
pvalue > alpha : 가설 채택 
pvalue < alpha : 가설 기각 
'''

if pvalue > alpha :
    print('정규분포와 차이가 없다.') # H0 채택 
else :
    print('정규분포와 차이가 있다.') # H1 채택(H0 기각) 
    
# [해설] 확률변수 X는 정규분포라 할 수있다. 


#######################################
## 정규분포와 표준정규분포
#######################################

# 단계1. 모집단에서 정규분포 추출   
'''
성인여성(19~24세)의 키는 평균이 162cm, 표준편차가 5cm인 정규분포
'''
np.random.seed(45)


# 모평균과 모표준편차 설정
mu = 162  # 평균키 
sigma = 5 # 표준편차 


# 정규분포에서 표본추출 : 확률변수 X : N(162, 5^2)
X = stats.norm.rvs(loc=mu, scale=sigma, size=1000) # 1000명  


# 단계2. 확률밀도함수(pdf)    

# 밀도곡선을 위한 벡터 자료   
line = np.linspace(min(X), max(X), 100)

# 히스토그램 : 단위(밀도)
plt.hist(X, bins='auto', density=True)  
plt.plot(line, stats.norm.pdf(line, mu, sigma), color='red') 
plt.show()


# 단계3. 정규성 검정 
statistic, pvalue = stats.shapiro(X)
print('검정통계량 = ', statistic) 
# 검정통계량 =  0.9991155044679678
print('p-value = ', pvalue) 
# p-value =  0.9258320244503772

# [해설] 확률변수 X는 정규분포라 할 수있다. 


# 단계4. 표준정규분포 : N(0, 1) 
sten_norm = (X - mu) / sigma # z = (X - 모평균) / 모표준편차 

# 밀도곡선을 위한 벡터 자료   
line = np.linspace(min(sten_norm), max(sten_norm), 100)

# 히스토그램 : 단위(밀도)
plt.hist(sten_norm, bins='auto', density=True)  
plt.plot(line, stats.norm.pdf(line, 0, 1), color='red') 
plt.show()

'''
가정 : ppt.31
   성인여성(19~24세)의 키는 평균이 162cm, 표준편차가 5cm인 정규분포를 
   따른다고 한다. 성인 여자의 키가 160cm 이하인 비율은 얼마일까?
'''

# 변수 설정 
x = 160
mu = 162
sigma = 5

# 표준화 
z = (x - mu) / sigma # -0.4
# z값에 대한 확률 : z분포표 이용 
p = 0.1554 # p(0 < z < 0.4) = 0.1554

# 160cm 이하 비율 : 좌우 대칭분포=0.5+0.5=1
result = 0.5 - p # 0.3446
print('160cm 이하인 비율 =', result * 100) # 160cm 이하인 비율 = 34.46


####################################################################################################

"""
2. 이산확률분포와 이항 검정 
  - 이산확률분포 : 베르누이분포, 이항분포, 포아송분포 등  
  - 이항분포와 검정 
"""

from scipy import stats # 확률분포 + 검정
import numpy as np # 성공횟수


################################
### 이항분포와 검정
################################
'''
 - 이항분포 : 2가지 범주(성공 or 실패)를 갖는 이산확률분포
 - 베르누이시행 : 이항변수(성공=1 or 실패=0)에서 독립시행 1회  
 - 베르누이분포 : 베르누이시행(독립시행 1회)으로 추출된 확률분포   
 - 이항분포 : 베루누이시행 n번으로 추출된 확률분포
   베르누이분포 : B(p)
   이항분포 : B(n, p)
'''
 

# 단계1. 표본 추출(random sampling) 

# 1) 동전 확률실험 : 베르누이분포 모집단에서 표본 추출
sample1 = stats.bernoulli.rvs(p=0.5, size=10) # B(p) 
sample1 # [0, 0, 0, 0, 1, 0, 0, 0, 1, 1]

# 2) 동전 확률실험 : 이항분포 모집단에서 표본 추출 
sample2 = stats.binom.rvs(n=1, p=0.5, size=10) # 독립시행=1회 
# [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]

sample3 = stats.binom.rvs(n=5, p=0.5, size=10) # 독립시행=5회 
# [2, 4, 2, 1, 3, 3, 3, 0, 3, 4]


# [문제] 주사위 확률실험 : 베르누이 독립시행 10회와 성공확률 1/6을 갖는 50개 표본 추출하기  
sample4 = stats.binom.rvs(n=10, p=1/6, size=50)
sample4


# 단계2. 이항검정(binom test) : 이항분포에 대한 가설검정 
'''
연구환경 : 게임에 이길 확률(p)이 40%이고, 게임의 시행회수가 50 일 때 95% 신뢰수준에서 검정 

귀무가설(H0) : 게임에 이길 확률(p)는 40%와 차이가 없다.(p = 40%)
대립가설(H1) : 게임에 이길 확률(p)는 40%와 차이가 있다.(p != 40%)
'''

np.random.seed(123) # 동일한 표본 추출  


# 1) 베르누이분포 : B(1, p)에서 표본추출(100개) 
p = 0.4 # 모수(p) : 성공확률 

# 베르누이분포 표본 추출 
binom_sample = stats.binom.rvs(n=1, p=p, size=50)
binom_sample
'''
[1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
1, 0, 0, 1, 1, 1]
'''

# 2) 성공횟수 반환 : zero 제외  
print('binom 성공회수 =', np.count_nonzero(binom_sample)) 
# binom 성공회수 = 23(27실패)


# 3) 유의확률 구하기  
k = np.count_nonzero(binom_sample) # 성공(1) 횟수 

# 4) 가설검정 
## 이항검정 수행 : 양측검정   
result = stats.binomtest(k=k, n=50, p=0.4, 
                         alternative='two-sided') 
'''
k : 성공횟수 
n : 시행횟수 
p : 성공확률(모수)
alternative : 검정방식(양측 or 단측)
'''
pvalue = result.pvalue # 0.39007355794671095 : 귀무가설 지지 확률 

# 이항검정 결과     
alpha = 0.05 # 100% - 5% = 95% : 95% 신뢰수준  

if pvalue > alpha :  
    print(f"p-value({pvalue}) : 게임에 이길 성공률 40%와 차이가 없다.")
else:
    print(f"p-value({pvalue}) : 게임에 이길 성공률 40%와 차이가 있다.")



######################### 
# 이항검정 적용 사례   
#########################
'''
연구환경 :  
  150명의 합격자 중에서 남자 합격자가 62명일 때 99% 신뢰수준에서 
  남여 합격률에 차이가 있다고 할수 있는가?

귀무가설(H0) : 남여 합격률에 차이가 없다.(p = 0.5)
대립가설(H0) : 남여 합격률에 차이가 있다.(p != 0.5)
'''


# 1) 이항검정 수행  
k = 62 # 성공회수 
result = stats.binomtest(k=k, n=150, p=0.5, alternative='two-sided')  
pvalue = result.pvalue # 0.04086849386649401 = 4% > 1%


# 2) 이항검정 결과   
print('## 이항 검정 ##')
alpha = 0.01 # 100% - 1% = 99% 신뢰수준(1-alpha)

if pvalue > alpha : # 유의확률 > 유의수준 
    print(f"p-value({pvalue}) >= 0.05 : 남여 합격률에 차이가 없다.")
else:
    print(f"p-value({pvalue}) < 0.05 : 남여 합격률에 차이가 있다.")
    
# 남여 합격률에 차이가 없다. : 귀무가설 채택  
    
alpha = 0.05 # 100% - 5% = 95% 신뢰수준(1-alpha) 

if pvalue > alpha : # 유의확률 > 유의수준 
    print(f"p-value({pvalue}) >= 0.05 : 남여 합격률에 차이가 없다.")
else:
    print(f"p-value({pvalue}) < 0.05 : 남여 합격률에 차이가 있다.")
 
# 남여 합격률에 차이가 있다. : 귀무가설 기각    
 
################################
### 단측검정으로 확인 
################################
# 귀무가설 : 남자합격자 > 여자합격자 : greater(>)
# 귀무가설 : 남자합격자 < 여자합격자 : less(<)

result = stats.binomtest(k=k, n=150, p=0.5, alternative='greater')
pvalue = result.pvalue

alpha = 0.05 

if pvalue > alpha :
    print('남자합격자 > 여자합격자')
else :
    print('남자합격자 < 여자합격자')


####################################################################################################

'''
t검정 : t 분포에 대한 가설검정  
  1. 단일표본 t검정 : 한 집단 평균차이 검정  
  2. 독립표본 t검정 : 두 집단 평균차이 검정
  3. 대응표본 t검정 : 대응 두 집단차이 검정 
'''

from scipy import stats # test
import numpy as np # sampling
import pandas as pd # csv file read


### 1. 단일표본 t검정 : 한 집단 평균차이 검정(모평균검정)   

# 대립가설(H1) : 모평균(mu) ≠ 174 -> 양측검정 
# 귀무가설(H0) : 모평균(mu) = 174 

# 남자 평균 키  170cm ~ 180cm -> 29명 표본추출 
sample_data = np.random.uniform(170,180, size=29) 
print(sample_data)

# 기술통계 
print('평균 키 =', sample_data.mean()) 
# 평균 키 = 174.78984854539283

'''
대립가설 기준 가설유형 결정
귀무가설 기준 가설검정 수행  
'''
# 단일집단 평균차이 검정 
one_group_test = stats.ttest_1samp(sample_data, 174, 
                                   alternative='two-sided') 
one_group_test
'''
statistic=1.7404543796317171, : 검정통계량 
pvalue=0.09276039858082241, : 유의확률  
df=28 : 자유도(n-1)
'''

print('t검정 통계량 = %.3f, pvalue = %.5f'%(one_group_test))
# t검정 통계량 = 1.740, pvalue = 0.09276

# 가설검정 
pvalue = 0.09276 # 유의확률 
alpha = 0.05 # 5% : 유의수준(알파)

if pvalue > alpha :
   print('모평균(mu) = 174') 
else :
   print('모평균(mu) ≠ 174')  

# [해설] 귀무가설 채택 : 모평균 174와 차이가 없다. 


### 2. 독립표본 t검정 :  두 집단 평균차이 검정

# 대립가설(H1) : 남자평균점수 < 여자평균점수 -> 단측검정 
# 귀무가설(H0) : 남여 평균 점수에 차이가 없다.

np.random.seed(36)
male_score = np.random.uniform(45, 95, size=30) # 남성 : a집단 
female_score = np.random.uniform(50, 100, size=30) # 여성 : b집단 

help(stats.ttest_ind)
'''
ttest_ind(a, b, alternative='two-sided')
{'two-sided', 'less(<)' : 좌측검정, 'greater(>) : 우측검정'}
'''
          
two_sample = stats.ttest_ind(male_score, female_score, 
                             alternative='less') # 단측검정 
print(two_sample)
'''
statistic=-2.1388869595347586, 
pvalue=0.018333434647486622, 
df=58.0
'''
print('두 집단 평균 차이 검정 = %.3f, pvalue = %.3f'%(two_sample))

pvalue = two_sample.pvalue
alpha = 0.05 

if pvalue > alpha :
    print('남여 평균 점수에 차이가 없다.')
else : 
    print('남자평균점수 < 여자평균점수')
    
# 남자평균점수 < 여자평균점수    

# 기술통계 : 평균 
male_score.mean() #  65.30708482736435
female_score.mean() # 72.72022332850773


# file 자료 이용 : 교육방법에 따른 실기점수의 평균차이 검정  

# 대립가설(H1) : 교육방법에 따른 실기점수의 평균에 차이가 있다.
# 귀무가설(H0) : 교육방법에 따른 실기점수의 평균에 차이가 없다.

sample = pd.read_csv(r'C:\ITWILL\4_Python_ML\data\two_sample.csv')
print(sample.info())

two_df = sample[['method', 'score']]
print(two_df)

two_df.isnull().sum()
'''
method     0
score     60
'''

# NA -> 평균 대체 
two_df['score'] = two_df.score.fillna(two_df.score.mean())

# 교육방법 기준 subset
method1 = two_df[two_df.method==1] # 방법1
method2 = two_df[two_df.method==2] # 방법2


# score 칼럼 추출 
score1 = method1.score
score2 = method2.score


# 두 집단 평균차이 검정 
two_sample = stats.ttest_ind(score1, score2) # 양측검정 
print(two_sample)
'''
statistic=-0.7095454005654631, 
pvalue=0.47868063184799514, 
df=238.0)
'''    

### 3. 대응표본 t검정 : 대응 두 집단 평균차이 검정

# 대립가설(H1) : 복용전과 복용후 몸무게 차이가 0 보다 크다.(복용전 몸무게 > 복용후 몸무게)
# 귀무가설(H0) : 복용전과 복용후 몸무게 차이에 변화가 없다.

before = np.random.randint(60, 65, size=30)  
after = np.random.randint(59, 64,  size=30) 

# 차이 = 복용전 - 복용후
before.mean() - after.mean() # 1.06666666666667
62.06 - 61 
'''
차이 > 0 : 식품에 효과 있음 
차이 < 0 : 식품에 효과 없음 
'''
paired_sample = stats.ttest_rel(before, after) # 양측검정 
print(paired_sample)
'''
statistic=2.6795305457936824, 
pvalue=0.012023383776261073, : 가설 기각 
df=29
'''
print('t검정 통계량 = %.5f, pvalue = %.5f'%paired_sample)

'''
ttest_ind(a, b, alternative='two-sided')
{'two-sided', 'less(<)' : 좌측검정, 'greater(>) : 우측검정'}
'''

# 단측검정 : 복용전 몸무게 > 복용후 몸무게
paired_sample = stats.ttest_rel(before, after, alternative='greater')
paired_sample
'''
statistic=2.6795305457936824, 
pvalue=0.0060116918881305366, : 가설 기각 
df=29
'''


####################################################################################################

"""
분산분석(ANOVA)
 - 세  집단 이상의 평균차이 검정(분산 이용) 
"""

import numpy as np
import pandas as pd

from scipy import stats # 일원분산분석(One-way_ANOVA) 
from statsmodels.formula.api import ols # 이원분산분석모델 생성  
import statsmodels.api as sm # 이원분산분석(Two_way_ANOVA) 


### 1. 일원분산분석(One-way_ANOVA) : y ~ x 
'''
가정 : 세 개의 그룹 (group1, group2, group3)을 생성하고, 이들 간의 평균 차이를 비교한다. 
기본가설 : 세 그룹의 평균은 차이가 없다.
'''

# 세 그룹의 만족도 데이터셋 
group1 = np.array([4, 6, 8, 10, 8])
group2 = np.array([2, 5, 7, 9, 4])
group3 = np.array([1, 3, 5, 2, 4])

 
# 귀무가설(H0) : 세 그룹의 만족도의 평균은 차이가 없다.
# 대립가설(H1) : 적어도 한 집단 이상에서 만족도의 평균에 차이가 있다. 

# 일원분산분석
f_statistic, p_value = stats.f_oneway(group1, group2, group3) 

# 결과 출력
print("F-statistic:", f_statistic) # 4.4399999999999995
print("p-value:", p_value) # 0.03603333923006122

# 기술통계 : 사후검정 
group1.mean() # 7.2
group2.mean() # 5.4
group3.mean() # 3.0



### 2. 이원분산분석(Two_way_ANOVA) : y ~ x1 + x2

'''
가정 : 약품종류 : A, B, C의 그룹에서 1시간, 2시간 간격으로 측정한 결과에 대한  
       반응 시간에 차이를 검정한다.
독립변수1: 약품종류(A, B, C)
독립변수2: 측정시간(1시간, 2시간)
종속변수: 반응시간
'''

# 귀무가설(H0) : 약품종류와 측정시간은 반응시간에 대해서 집단간 평균의 차이는 없다.  

data = pd.DataFrame({
    'type': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'], # 약품종류
    'time': [1, 2, 1, 2, 1, 2, 1, 2, 2],  # 측정시간 
    'retime': [10, 15, 9, 8, 14, 11, 11, 16, 20] # 반응시간 
})



# 이원분산분석 모델 생성 
model = ols('retime ~ type + time', data=data).fit()

# 이원분산분석 
anova_table = sm.stats.anova_lm(model)

# 결과 출력
print(anova_table)
'''
           df     sum_sq    mean_sq         F    PR(>F)
type      2.0  40.666667  20.333333  1.561433  0.297270  > 0.05
time      1.0  14.222222  14.222222  1.092150  0.343864  > 0.05
Residual  5.0  65.111111  13.022222       NaN       NaN
'''
# [해설] 약품종류와 측정시간은 반응시간에 대해서 집단간 평균의 차이는 없다. 


####################################################################################################

'''
 카이제곱 검정(chisquare test) 
  - 확률변수의 적합성 검정 - 일원  
  - 두 집단변수 간의 독립성 검정 - 이원 
  - 검정통계량(기대비율) = sum( (관측값 - 기댓값)**2 / 기댓값 )
'''

from scipy import stats # 확률분포 검정 


### 1. 일원 chi-square(1개 변수 이용) : 적합성 검정 
'''
 귀무가설 : 관측치와 기대치는 차이가 없다.(게임에 적합하다.)
 대립가설 : 관측치와 기대치는 차이가 있다.(게임에 적합하지않다.) 
'''

# 주사위 적합성 검정 : 60번 시행 
real_data = [4, 6, 17, 16, 8, 9] # 관측값 - 관측도수 
exp_data = [10,10,10,10,10,10] # 기대값 - 기대도수 
chis = stats.chisquare(real_data, exp_data)
print(chis)
print('statistic = %.3f, pvalue = %.3f'%(chis)) # statistic = 14.200, pvalue = 0.014
# statistic = 14.200, pvalue = 0.014 < alpha = 0.05 : 가설기각 
# 검정통계량(statistic) : -1.96 ~ 1.96(채택역)

# [해설] 유의미한 수준(5%)에서 주사위는 게임에 적합하다고 볼 수 없다.


### 2. 이원 chi-square(2개 변수 이용) : 교차행렬의 관측값과 기대값으로 검정
'''
 귀무가설 : 교육수준과 흡연율 간에 관련성이 없다.(독립적이다.)
 대립가설 : 교육수준과 흡연율 간에 관련성이 있다.(독립적이지 않다.)
'''

# 파일 가져오기
import pandas as pd

path = r'C:\ITWILL\4_Python_ML\data'
smoke = pd.read_csv(path + "/smoke.csv")
smoke.info()

# <단계 1> 변수 선택 
print(smoke)# education, smoking 변수
education = smoke.education 
smoking = smoke.smoking 


# <단계 2> 교차분할표 
tab = pd.crosstab(index=education, columns=smoking)
print(tab) # 관측값 
'''
smoking     1   2   3
education            
1          51  92  68
2          22  21   9
3          43  28  21
'''

# <단계3> 카이제곱 검정 : 교차분할표 이용 
chi2, pvalue, df, evalue = stats.chi2_contingency(observed= tab)  

# chi2 검정통계량, 유의확률, 자유도, 기대값  
print('chi2 = %.6f, pvalue = %.6f, d.f = %d'%(chi2, pvalue, df))
# chi2 = 18.910916, pvalue = 0.000818, d.f = 4


# <단계4> 기대값 
print(evalue)
'''
smoking     1   2   3
education            
1          51  92  68
2          22  21   9
3          43  28  21
chi2 = 18.910916, pvalue = 0.000818, d.f = 4
[[68.94647887 83.8056338  58.24788732]
 [16.9915493  20.65352113 14.35492958]
 [30.06197183 36.54084507 25.3971831 ]]
'''

#############################################
# 성별과 흡연 간의 독립성 검정 example 
#############################################
'''
 귀무가설 : 성별과 흡연유무 간에 관련성이 없다.
 대립가설 : 성별과 흡연유무 간에 관련성이 있다.
'''
import seaborn as sn
import pandas as pd

# <단계1> titanic dataset load 
tips = sn.load_dataset('tips')
print(tips.info())

# <단계2> 교차분할표 
tab = pd.crosstab(index=tips.sex, columns=tips.smoker)
print(tab)
'''
smoker  Yes  No
sex            
Male     60  97
Female   33  54
'''

# <단계3> 카이제곱 검정 
chi2, pvalue, df, evalue = stats.chi2_contingency(observed= tab)
print(chi2, pvalue, df)  # 0.0 1.0 1  

print('기대빈도')
print(evalue)
'''
[[59.84016393 97.15983607]
 [33.15983607 53.84016393]]
'''

####################################################################################################

'''
공분산 vs 상관계수 
 
1. 공분산 : 두 확률변수 간의 분산(평균에서 퍼짐 정도)를 나타내는 통계 
  - 식 : Cov(X,Y) = sum( (X-x_bar) * (Y-y_bar) ) / n
 
  - Cov(X, Y) > 0 : X가 증가할 때 Y도 증가
  - Cov(X, Y) < 0 : X가 증가할 때 Y는 감소
  - Cov(X, Y) = 0 : 두 변수는 선형관계 아님(서로 독립적 관계) 
  - 문제점 : 값이 큰 변수에 영향을 받는다.(값 큰 변수가 상관성 높음)
    
2. 상관계수 : 공분산을 각각의 표준편차로 나눈어 정규화한 통계
   - 공분산 문제점 해결 
   - 부호는 공분산과 동일, 값은 절대값 1을 넘지 않음(-1 ~ 1)    
   - 식 : Corr(X, Y) = Cov(X,Y) / std(X) * std(Y)
'''

import pandas as pd 
score_iq = pd.read_csv(r'c:/itwill/4_python_ml/data/score_iq.csv')
print(score_iq)
'''
score iq    academy
90   140        2
75   125        1
'''

# 1. 피어슨 상관계수 행렬 
corr = score_iq.corr(method='pearson')
print(corr)
 
corr['score']
'''
score      1.000000
iq         0.882220
academy    0.896265
'''

# 2. 공분산 행렬 
cov = score_iq.cov()
print(cov)
cov['score']
'''
score      42.968412
iq         51.337539
academy     7.119911
'''

# 3. 공분산 vs 상관계수 식 적용 

#  1) 공분산 : Cov(X, Y) = sum( (X-x_bar) * (Y-y_bar) ) / n
X = score_iq['score']
Y = score_iq['iq']

# 표본평균 
x_bar = X.mean()
y_bar = Y.mean()

# 표본의 공분산 
Cov = sum((X - x_bar)  * (Y - y_bar)) / (len(X)-1)
print('Cov =', Cov) 


# 2) 상관계수 : Corr(X, Y) = Cov(X,Y) / std(X) * std(Y)
stdX = X.std()
stdY = Y.std()

Corr = Cov / (stdX * stdY)
print('Corr =', Corr) # Corr = 0.8822203446134699

####################################################################################################

"""
scipy 패키지 이용 
 1. 단순선형회귀분석 : Y ~ X
 2. 다중선형회귀분석 : Y ~ X1 + X2 + X3
"""

from scipy import stats
import pandas as pd

#귀무가설(H0) : iq는 score에 영향을 미치지 않는다.(기각)
#대립가설(H1) : iq는 score에 영향을 미친다.(채택)

score_iq = pd.read_csv('c:/itwill/4_python_ml/data/score_iq.csv')
score_iq.info()

# 1. 단순선형회귀분석 
'''
x -> y
'''

# 1) 변수 생성 
x = score_iq['iq'] # 독립변수(설명변수) 
y = score_iq['score'] # 종속변수(반응변수) 

# 2) model 생성 
model = stats.linregress(x, y)
print(model)
'''
LinregressResult(
    slope=0.6514309527270075, : x 기울기 
    intercept=-2.8564471221974657, : y 절편 
    rvalue=0.8822203446134699, : 설명력
    pvalue=2.8476895206683644e-50, : x의 유의성 검정  
    stderr=0.028577934409305443) : 표준오차 
'''

a = model.slope # x 기울기
b = model.intercept # y 절편 

# 회귀방정식 -> y 예측치 
X = 140; Y = 90 # 1개 관측치 

y_pred = (X*a) + b # 회귀방정식 
print(y_pred) # 88.34388625958358

err = Y - y_pred
print('err=', err) # err= 1.6561137404164157

# 전체 관측치 대상 
len(x) # 150
y_pred = (x*a) + b # 예측치 
len(y_pred) # 150

# 관측치 vs 예측치 
print('관측치 평균 : ', y.mean())
print('예측치 평균 : ', y_pred.mean())

print(y[:10])
print(y_pred[:10])


# 2. 회귀모델 시각화 
import matplotlib.pyplot as plt

# 산점도 
plt.plot(score_iq['iq'], score_iq['score'], 'b.')
# 회귀선 
plt.plot(score_iq['iq'], y_pred, 'r.-')
plt.title('line regression') # 제목 
plt.legend(['x y scatter', 'line regression']) # 범례 
plt.show()



# 3. 다중선형회귀분석 : formula 형식 
from statsmodels.formula.api import ols

'''
기본가설 : 모든 독립변수는 종속변수에 영향이 없다.
'''

# 상관계수 행렬 
corr = score_iq.corr()
print(corr['score'])


obj = ols(formula='score ~ iq + academy + tv', data = score_iq)
model = obj.fit()

dir(model)


# 회귀계수값 반환 
print('회귀 계수값\n%s'%(model.params))
'''
Intercept    24.722251 : 절편 
iq            0.374196 : x1기울기 
academy       3.208802 : x2기울기
tv            0.192573 : x3기울기
'''

# model의 적합치 
print('model 적합치 :', model.fittedvalues)

# 회귀분석 결과 제공  
print(model.summary()) 
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  score   R-squared:                       0.946
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     860.1
Date:                Wed, 01 May 2024   Prob (F-statistic):           1.50e-92
Time:                        12:08:37   Log-Likelihood:                -274.84
No. Observations:                 150   AIC:                             557.7
Df Residuals:                     146   BIC:                             569.7
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     24.7223      2.332     10.602      0.000      20.114      29.331
iq             0.3742      0.020     19.109      0.000       0.335       0.413
academy        3.2088      0.367      8.733      0.000       2.483       3.935
tv             0.1926      0.303      0.636      0.526      -0.406       0.791
==============================================================================
Omnibus:                       36.802   Durbin-Watson:                   1.905
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               57.833
Skew:                           1.252   Prob(JB):                     2.77e-13
Kurtosis:                       4.728   Cond. No.                     2.32e+03
==============================================================================

1. 모형에 유의성 검정 : Prob (F-statistic):  1.50e-92
2. 모형에 대한 설명력 : Adj. R-squared:  0.945
3. X변수에 대한 유의성 검정 : t   ->   P>|t|
'''

