---
layout: single
title:  "성공"
categories: jupyter
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

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
X = df[['3qtr_ptsgap','pts_home','pts_away','team_id_home','pts_qtr1_home','pts_qtr2_home','pts_qtr3_home','pts_qtr1_away','pts_qtr2_away','pts_qtr3_away']]
y = df['wl_home']
```


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
((4412, 10), (1103, 10))
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
Model accuracy score with 10 decision-trees : 0.5286
</pre>

```python
#의사결정나무 100개 지정, 학습
rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_100.fit(X_train, y_train)
```

<style>#sk-container-id-11 {color: black;background-color: white;}#sk-container-id-11 pre{padding: 0;}#sk-container-id-11 div.sk-toggleable {background-color: white;}#sk-container-id-11 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-11 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-11 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-11 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-11 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-11 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-11 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-11 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-11 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-11 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-11 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-11 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-11 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-11 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-11 div.sk-item {position: relative;z-index: 1;}#sk-container-id-11 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-11 div.sk-item::before, #sk-container-id-11 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-11 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-11 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-11 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-11 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-11 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-11 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-11 div.sk-label-container {text-align: center;}#sk-container-id-11 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-11 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-11" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" checked><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=0)</pre></div></div></div></div></div>



```python
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0,min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
```

<style>#sk-container-id-12 {color: black;background-color: white;}#sk-container-id-12 pre{padding: 0;}#sk-container-id-12 div.sk-toggleable {background-color: white;}#sk-container-id-12 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-12 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-12 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-12 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-12 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-12 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-12 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-12 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-12 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-12 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-12 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-12 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-12 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-12 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-12 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-12 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-12 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-12 div.sk-item {position: relative;z-index: 1;}#sk-container-id-12 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-12 div.sk-item::before, #sk-container-id-12 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-12 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-12 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-12 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-12 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-12 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-12 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-12 div.sk-label-container {text-align: center;}#sk-container-id-12 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-12 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-12" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(max_features=&#x27;auto&#x27;, random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" checked><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_features=&#x27;auto&#x27;, random_state=0)</pre></div></div></div></div></div>



```python
# Predict on the test set results
y_pred_100 = rfc_100.predict(X_test)
```


```python
#accuracy_score 확인
print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))
```

<pre>
Model accuracy score with 100 decision-trees : 0.5286
</pre>

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_100))
```

<pre>
              precision    recall  f1-score   support

           0       0.43      0.28      0.34       473
           1       0.57      0.71      0.63       630

    accuracy                           0.53      1103
   macro avg       0.50      0.50      0.49      1103
weighted avg       0.51      0.53      0.51      1103

</pre>

```python
#Feacher Importance 확인
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores[:10]
```

<pre>
pts_home         0.123189
pts_away         0.118315
pts_qtr2_home    0.109243
3qtr_ptsgap      0.109175
pts_qtr3_away    0.108865
pts_qtr3_home    0.108806
pts_qtr1_home    0.108043
pts_qtr2_away    0.107562
pts_qtr1_away    0.106801
team_id_home     0.000000
dtype: float64
</pre>

```python
```


```python
```
