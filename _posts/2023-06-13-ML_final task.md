---
layout: single
title:  "NBA 우승팀 분석"
categories: jupyter
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

# 0. 사전 설명 
## -1. 유튜브 영상 
이 블로그에 대한 내용이 설명된 유튜브 링크: 

## -2. 멤버들 
오준혁, 기계공학부, dhwnsgurm@naver.com \
이한결, 경영학부,  hglee412@gmail.com \
이우섭, 용합전자공학부, lws011106@hanyang.ac.kr \
최정웅, 건설환경공학과, jungwoong.choi17@gmail.com <br/><br/>


# 1. Proposal (option A)
## -1. 많은 스포츠들 중 농구를 선택한 이유 (다양한 변수, 확실하지 않은 지표)
 &nbsp; 세상에는 축구, 농구, 야구, 배구 등 다양한 스포츠가 존재한다. 그리고 이러한 스포츠들은 각각의 방식, 지표를 통해 경기의 승부나 시즌의 순위 등을 예측한다. 여기서 농구는 다른 종목들에 비해 다양한 변수를 가지고 있다. 농구 경기의 승패를 가르는 변수에는 쿼터당 점수, 반칙의 갯수, 연승부터 시작해서 경기장의 상태, 각 선수들의 지표까지 존재하기 때문이다. 심지어 5명이 뛰는 경기이기에 선수 한명 한명의 컨디션 또한 중요한 변수로 작용하게 된다. 이렇게 다양한 변수가 있지만, 다른 스포츠에 비해 확실한 지표가 존재하진 않는다. 예를들어 축구에서는 승률, 점유율, 각 선수들의 활동영역 등 대표적으로 중요시하는 지표들이 존재한다. 반면 농구는 딱 떠오르는 대표적인 지표가 존재하지 않고, 다양한 변수로 인해 승패를 예측하기 어렵기에 AI+X딥러닝 프로젝트에 적합하다고 생각 했다. 지표를 학습하고 테스트 하는 데이터는 세계적인 농구 리그인 NBA (National Basketball Association)의 데이터를 사용할 것이다. 
 
## -2. 승률 분석에 사용하는 지표 설정의 중요성 
 &nbsp; 현재 농구 리그의 승패 예측에 사용되는 지표 중 가장 대표적인 것은 BPI이다. BPI는 Basketball Power Index(농구 파워지수)의 약자로, 세계 최대 스포트 사이트인 ESPN에서 독자적으로 개발하여 사용 중인 지표이다. 세계적인 사이트의 지표이기에 그 정확도 또한 높을 것이라는 믿음과 달리 최근 경기에 대한 예측에 논란이 있었다. 보스턴 셀틱스와 마이애미 히트의 7판 4선 경기에서 마이애미가 3승 0패라는 압도적인 결과를 내고 있음에도 불구하고 보스턴이 72%로 이길 것이라는 예측을 내놓았기 때문이다. BPI는 각 팀의 공격력, 수비력 등을 전문가들이 "주관적"으로 만든 지표이기에 이렇게 낮은 정확도에 대한 논란이 있었던 것이다.\
 &nbsp; 아래는 ESPN에서 올해 2022 -2023 NBA시즌의 순위를 예측한 자료이다. 

![Alt text](https://github.com/WhenIwas20/WhenIwas20.github.io/assets/132420091/04fc4fc7-8470-4651-af0d-9a7159e972d6)  


  &nbsp; 실제 순위와 비교해보면 정답률은 6.67% (15개 순위 중 하나의 순위만 맞춤)임을 확인 할 수 있다. 이러한 자료들을 통해 현재 사용하고 있는 지표는 정확한 예측으로 이어지지 않는다는 것을 확인할 수 있고, 그렇기에 다양한 변수를 살펴보며 순위를 정확하게 예측할 수 있는 지표를 만드는 것을 목표로 하여 프로젝트를 진행하였다. <br/><br/>


# 2. Datesets(nba 팀들의 승률 분석)
###  변수 설정 

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>



```python
#작업에 필요한 라이브러리 불러오기
from IPython.display import display, HTML
display(HTML("<style>.container {width:90% !important;}</style>"))

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("nbadatafinal_realreal.csv")
df.shape
```

<style>.container {width:90% !important;}</style>


<pre>
(5515, 23)
</pre>

```python
#연속형변수와 범주형변수 구분
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5515 entries, 0 to 5514
Data columns (total 23 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   game_date_est           5515 non-null   object
 1   game_id                 5515 non-null   int64 
 2   team_id_home            5515 non-null   int64 
 3   team_abbreviation_home  5515 non-null   object
 4   team_city_name_home     5515 non-null   object
 5   team_nickname_home      5515 non-null   object
 6   qtr1+2+3_home           5515 non-null   int64 
 7   qtr1+2+3_away           5515 non-null   int64 
 8   3qtr_ptsgap             5515 non-null   int64 
 9   pts_home                5515 non-null   int64 
 10  pts_away                5515 non-null   int64 
 11  wl_home                 5515 non-null   int64 
 12  wl_away                 5515 non-null   int64 
 13  team_id_away            5515 non-null   int64 
 14  team_abbreviation_away  5515 non-null   object
 15  team_city_name_away     5515 non-null   object
 16  team_nickname_away      5515 non-null   object
 17  pts_qtr1_home           5515 non-null   int64 
 18  pts_qtr2_home           5515 non-null   int64 
 19  pts_qtr3_home           5515 non-null   int64 
 20  pts_qtr1_away           5515 non-null   int64 
 21  pts_qtr2_away           5515 non-null   int64 
 22  pts_qtr3_away           5515 non-null   int64 
dtypes: int64(16), object(7)
memory usage: 991.1+ KB
</pre>

```python
#연속형변수 16개
df.describe().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>game_id</th>
      <td>5515.0</td>
      <td>2.199629e+07</td>
      <td>141849.318029</td>
      <td>2.180000e+07</td>
      <td>2.190017e+07</td>
      <td>2.200052e+07</td>
      <td>2.210085e+07</td>
      <td>2.220102e+07</td>
    </tr>
    <tr>
      <th>team_id_home</th>
      <td>5515.0</td>
      <td>1.610613e+09</td>
      <td>8.651740</td>
      <td>1.610613e+09</td>
      <td>1.610613e+09</td>
      <td>1.610613e+09</td>
      <td>1.610613e+09</td>
      <td>1.610613e+09</td>
    </tr>
    <tr>
      <th>qtr1+2+3_home</th>
      <td>5515.0</td>
      <td>8.419529e+01</td>
      <td>10.640945</td>
      <td>4.100000e+01</td>
      <td>7.700000e+01</td>
      <td>8.400000e+01</td>
      <td>9.100000e+01</td>
      <td>1.270000e+02</td>
    </tr>
    <tr>
      <th>qtr1+2+3_away</th>
      <td>5515.0</td>
      <td>8.433436e+01</td>
      <td>10.655974</td>
      <td>3.800000e+01</td>
      <td>7.700000e+01</td>
      <td>8.400000e+01</td>
      <td>9.100000e+01</td>
      <td>1.230000e+02</td>
    </tr>
    <tr>
      <th>3qtr_ptsgap</th>
      <td>5515.0</td>
      <td>1.081795e+01</td>
      <td>8.408086</td>
      <td>0.000000e+00</td>
      <td>4.000000e+00</td>
      <td>9.000000e+00</td>
      <td>1.500000e+01</td>
      <td>5.600000e+01</td>
    </tr>
    <tr>
      <th>pts_home</th>
      <td>5515.0</td>
      <td>1.117039e+02</td>
      <td>12.591252</td>
      <td>4.100000e+01</td>
      <td>1.030000e+02</td>
      <td>1.120000e+02</td>
      <td>1.200000e+02</td>
      <td>1.750000e+02</td>
    </tr>
    <tr>
      <th>pts_away</th>
      <td>5515.0</td>
      <td>1.120892e+02</td>
      <td>12.576701</td>
      <td>3.800000e+01</td>
      <td>1.040000e+02</td>
      <td>1.120000e+02</td>
      <td>1.200000e+02</td>
      <td>1.760000e+02</td>
    </tr>
    <tr>
      <th>wl_home</th>
      <td>5515.0</td>
      <td>5.655485e-01</td>
      <td>0.495730</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>wl_away</th>
      <td>5515.0</td>
      <td>4.344515e-01</td>
      <td>0.495730</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>team_id_away</th>
      <td>5515.0</td>
      <td>1.610613e+09</td>
      <td>8.642790</td>
      <td>1.610613e+09</td>
      <td>1.610613e+09</td>
      <td>1.610613e+09</td>
      <td>1.610613e+09</td>
      <td>1.610613e+09</td>
    </tr>
    <tr>
      <th>pts_qtr1_home</th>
      <td>5515.0</td>
      <td>2.813672e+01</td>
      <td>6.038819</td>
      <td>8.000000e+00</td>
      <td>2.400000e+01</td>
      <td>2.800000e+01</td>
      <td>3.200000e+01</td>
      <td>5.100000e+01</td>
    </tr>
    <tr>
      <th>pts_qtr2_home</th>
      <td>5515.0</td>
      <td>2.822430e+01</td>
      <td>5.946496</td>
      <td>9.000000e+00</td>
      <td>2.400000e+01</td>
      <td>2.800000e+01</td>
      <td>3.200000e+01</td>
      <td>4.900000e+01</td>
    </tr>
    <tr>
      <th>pts_qtr3_home</th>
      <td>5515.0</td>
      <td>2.783427e+01</td>
      <td>6.067307</td>
      <td>0.000000e+00</td>
      <td>2.400000e+01</td>
      <td>2.800000e+01</td>
      <td>3.200000e+01</td>
      <td>5.500000e+01</td>
    </tr>
    <tr>
      <th>pts_qtr1_away</th>
      <td>5515.0</td>
      <td>2.831895e+01</td>
      <td>6.031684</td>
      <td>9.000000e+00</td>
      <td>2.400000e+01</td>
      <td>2.800000e+01</td>
      <td>3.200000e+01</td>
      <td>5.100000e+01</td>
    </tr>
    <tr>
      <th>pts_qtr2_away</th>
      <td>5515.0</td>
      <td>2.811224e+01</td>
      <td>5.908834</td>
      <td>9.000000e+00</td>
      <td>2.400000e+01</td>
      <td>2.800000e+01</td>
      <td>3.200000e+01</td>
      <td>5.000000e+01</td>
    </tr>
    <tr>
      <th>pts_qtr3_away</th>
      <td>5515.0</td>
      <td>2.790317e+01</td>
      <td>5.991263</td>
      <td>0.000000e+00</td>
      <td>2.400000e+01</td>
      <td>2.800000e+01</td>
      <td>3.200000e+01</td>
      <td>4.900000e+01</td>
    </tr>
  </tbody>
</table>
</div>



```python
categorical = [var for var in df.columns if df[var].dtype=='O']
df[categorical].head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_date_est</th>
      <th>team_abbreviation_home</th>
      <th>team_city_name_home</th>
      <th>team_nickname_home</th>
      <th>team_abbreviation_away</th>
      <th>team_city_name_away</th>
      <th>team_nickname_away</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-10-16 0:00</td>
      <td>BOS</td>
      <td>Boston</td>
      <td>Celtics</td>
      <td>PHI</td>
      <td>Philadelphia</td>
      <td>76ers</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-10-16 0:00</td>
      <td>GSW</td>
      <td>Golden State</td>
      <td>Warriors</td>
      <td>OKC</td>
      <td>Oklahoma City</td>
      <td>Thunder</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-10-17 0:00</td>
      <td>SAS</td>
      <td>San Antonio</td>
      <td>Spurs</td>
      <td>MIN</td>
      <td>Minnesota</td>
      <td>Timberwolves</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-10-17 0:00</td>
      <td>MEM</td>
      <td>Memphis</td>
      <td>Grizzlies</td>
      <td>IND</td>
      <td>Indiana</td>
      <td>Pacers</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-10-17 0:00</td>
      <td>NOP</td>
      <td>New Orleans</td>
      <td>Pelicans</td>
      <td>HOU</td>
      <td>Houston</td>
      <td>Rockets</td>
    </tr>
  </tbody>
</table>
</div>



```python
#범주형변수 결측치 확인
df[categorical].isnull().sum()
```

<pre>
game_date_est             0
team_abbreviation_home    0
team_city_name_home       0
team_nickname_home        0
team_abbreviation_away    0
team_city_name_away       0
team_nickname_away        0
dtype: int64
</pre>

```python
numerical = [var for var in df.columns if df[var].dtype !='O']
df[numerical].head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_id</th>
      <th>team_id_home</th>
      <th>qtr1+2+3_home</th>
      <th>qtr1+2+3_away</th>
      <th>3qtr_ptsgap</th>
      <th>pts_home</th>
      <th>pts_away</th>
      <th>wl_home</th>
      <th>wl_away</th>
      <th>team_id_away</th>
      <th>pts_qtr1_home</th>
      <th>pts_qtr2_home</th>
      <th>pts_qtr3_home</th>
      <th>pts_qtr1_away</th>
      <th>pts_qtr2_away</th>
      <th>pts_qtr3_away</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21800001</td>
      <td>1610612738</td>
      <td>77</td>
      <td>66</td>
      <td>11</td>
      <td>105</td>
      <td>87</td>
      <td>1</td>
      <td>0</td>
      <td>1610612755</td>
      <td>21</td>
      <td>26</td>
      <td>30</td>
      <td>21</td>
      <td>21</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21800002</td>
      <td>1610612744</td>
      <td>83</td>
      <td>79</td>
      <td>4</td>
      <td>108</td>
      <td>100</td>
      <td>1</td>
      <td>0</td>
      <td>1610612760</td>
      <td>31</td>
      <td>26</td>
      <td>26</td>
      <td>23</td>
      <td>24</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21800010</td>
      <td>1610612759</td>
      <td>81</td>
      <td>83</td>
      <td>2</td>
      <td>112</td>
      <td>108</td>
      <td>1</td>
      <td>0</td>
      <td>1610612750</td>
      <td>31</td>
      <td>25</td>
      <td>25</td>
      <td>23</td>
      <td>29</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21800005</td>
      <td>1610612763</td>
      <td>58</td>
      <td>76</td>
      <td>18</td>
      <td>83</td>
      <td>111</td>
      <td>1</td>
      <td>0</td>
      <td>1610612754</td>
      <td>16</td>
      <td>23</td>
      <td>19</td>
      <td>27</td>
      <td>29</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21800009</td>
      <td>1610612740</td>
      <td>101</td>
      <td>84</td>
      <td>17</td>
      <td>131</td>
      <td>112</td>
      <td>0</td>
      <td>1</td>
      <td>1610612745</td>
      <td>35</td>
      <td>36</td>
      <td>30</td>
      <td>29</td>
      <td>25</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



```python
#연속형변수 결측치 확인
df[numerical].isnull().sum()
```

<pre>
game_id          0
team_id_home     0
qtr1+2+3_home    0
qtr1+2+3_away    0
3qtr_ptsgap      0
pts_home         0
pts_away         0
wl_home          0
wl_away          0
team_id_away     0
pts_qtr1_home    0
pts_qtr2_home    0
pts_qtr3_home    0
pts_qtr1_away    0
pts_qtr2_away    0
pts_qtr3_away    0
dtype: int64
</pre>

```python
#모두 결측치가 존재하지 않기 때문에 추가처리없이 진행
#입력 변수(X)와 출력 변수(y)를 분리
columns_to_drop = ['game_date_est', 'team_city_name_home', 'team_nickname_home','team_city_name_away','team_nickname_away','wl_away','team_abbreviation_home','team_abbreviation_away']
X = df.drop(columns_to_drop, axis=1)
y = df['wl_home']
```


```python
# 3쿼터까지의 점수차 데이터와 승패 데이터의 관계에 따라 가중치를 부여
#3쿼터까지 10점차 이하인 경기에 대해 승리시 2의 가중치 패배시 1.5의 가중치
#10점차 이상이 난 경기에 대해서는 1의 가중치 부여
weights = []
for val1, val2 in zip(df['3qtr_ptsgap'], df['wl_home']):
    if val1 <= 10 and val2 == 1:
        weights.append(2.0)
    elif val1 <= 10 and val2 == 0:
        weights.append(1.5)
    else:
        weights.append(1.0)

model = RandomForestClassifier()
model.fit(X, y, sample_weight=weights)
```

<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier()</pre></div></div></div></div></div>



```python
#Train, Test data set 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
# check the shape of X_train and X_test
X_train.shape, X_test.shape
```

<pre>
((4412, 15), (1103, 15))
</pre>

```python
#랜덤포레스트 모델 학습
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
##10개의 의사결정 트리를 사용하여 예측하는 모델의 정확도 점수를 계산하고, 소수점 이하 4자리까지 출력한다. accuracy_score는 올바르게 예측된 라벨의 비율을 테스트 데이터싯의 전체 샘플 수로 나눈 값이다.
from sklearn.metrics import accuracy_score
print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
```

<pre>
Model accuracy score with 10 decision-trees : 1.0000
</pre>

```python
#의사결정나무 100개 지정, 학습
rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_100.fit(X_train, y_train)
```

<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=0)</pre></div></div></div></div></div>



```python
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
```

<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(max_features=&#x27;auto&#x27;, random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_features=&#x27;auto&#x27;, random_state=0)</pre></div></div></div></div></div>



```python
# Predict on the test set results
y_pred_100 = rfc_100.predict(X_test)
```


```python
#accuracy_score 확인
print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))
```

<pre>
Model accuracy score with 100 decision-trees : 1.0000
</pre>

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_100))
```

<pre>
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       473
           1       1.00      1.00      1.00       630

    accuracy                           1.00      1103
   macro avg       1.00      1.00      1.00      1103
weighted avg       1.00      1.00      1.00      1103

</pre>

```python
#Feacher Importance 확인
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores[:10]
```

<pre>
wl_home          0.941218
game_id          0.007372
pts_away         0.004968
qtr1+2+3_home    0.004899
3qtr_ptsgap      0.004826
pts_home         0.004752
pts_qtr2_away    0.004730
pts_qtr3_away    0.004696
qtr1+2+3_away    0.004628
pts_qtr1_away    0.004563
dtype: float64
</pre>

```python
```
## tqtq

# 3. Methodology
## -1. random forest


# 4. Conclusion
## 

# 5. Related Work 
## -1. 포제션 개념(possession concept)
 &nbsp;  포제션은 소유라는 뜻으로 직역된다. 이 개념이 스포트에서는 볼을 소유한 시간, 다시 말해 공격할 기회를 의미하고 승패를 결정하는데 관련이 있는 변수 중 하나로 연구되고 있다. 경희대학교에서는 한국의 농구 기록 관련 연구에서 다뤄지지 않았던 포제션의 개념을 적용하여 한국남자프로농구의 승률을 예측하는 연구를 진행하였다. 한국남자프로농구 1997-1998시즌부터 2016-2017시즌의 자료를 대상으로 선정, 분석하여 1차 자료 수집으로 얻은 각 팀의 기록에 NBA에서 사용하는 포제션 산출 공식을 적용하여 팀별 포제션당 팀 평균기록을 산출하였다. 이러한 데이터를 바탕으로 상관분석, 다중회귀분석, Fisher의 Z 변환 및 단계적 다중회귀분석을 실시하여 포제션당 팀 평균기록은 경기당 팀 평균기록보다 팀 승률에 대한 설명력이 더 크다는 것과 포제션당 팀 평균기록으로 팀 승률을 예측했을 때, 수비 리바운드, 2점슛 성공률, 스틸, 3점슛 성공률, 자유투시도, 턴오버, 공격 리바운드, 블록슛, 자유투 성공률, 어시스트 순으로 영향을 미치는 것으로 나타났다는 유의미한 결과를 얻었다. 


## 참고자료
https://www.kaggle.com/datasets/wyattowalsh/basketball
https://www.boston.com/sports/boston-celtics/2023/05/20/celtics-heat-stats-odds-probability-espn-bpi-65-percent-game-2/?amp=1
https://n.news.naver.com/sports/basketball/article/425/0000140467
https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002330790
https://m.riss.kr/search/detail/ssoSkipDetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=830f162886a6f9c7ffe0bdc3ef48d419

