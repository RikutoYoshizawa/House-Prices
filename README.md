# House-Prices
the 1st task for the HAIT intern matching event
実践課題 vol.1
AI_STANDARD 機械学習講座vol.1~ vol.5の学習のアウトプットとして、実践的な課題を解いて行きましょう。kaggleというデータサイエンスのコンペティションサイトにある、住宅価格予測のデータを使用した課題です。



kaggle: House Prices Competition link

データのダウンロード
２つのデータをダウンロードして、jupyterを起動しているディレクトリに保存して下さい(右クリックでダウンロードを選択できます)。

前処理済みデータ
正解データ



In [1]:

# モジュールのインポート
​
​import pandas as pd
​import numpy as  np
​import seaborn as sns
​import matplotlib.pyplot as plt
​% matplotlib inline
​
​data = pd.read_csv('house_price.csv')
​y = pd.read_csv('y.csv')
​問題１：dataの上から20行を表示してください。
​In [3]:
​
​data[1:21]
​Out[3]:
​LotFrontage    LotArea    Street    LotShape    Utilities    LandSlope    OverallQual    OverallCond    YearBuilt    YearRemodAdd    ...    SaleType_ConLw    SaleType_New    SaleType_Oth    SaleType_WD    SaleCondition_Abnorml    SaleCondition_AdjLand    SaleCondition_Alloca    SaleCondition_Family    SaleCondition_Normal    SaleCondition_Partial
​1    80    9.169623    1.098612    1.609438    1.609438    1.386294    6    2.197225    7.589336    7.589336    ...    0    0    0    1    0    0    0    0    1    0
​2    68    9.328212    1.098612    1.386294    1.609438    1.386294    7    1.791759    7.601902    7.602401    ...    0    0    0    1    0    0    0    0    1    0
​3    60    9.164401    1.098612    1.386294    1.609438    1.386294    7    1.791759    7.557995    7.586296    ...    0    0    0    1    1    0    0    0    0    0
​4    84    9.565284    1.098612    1.386294    1.609438    1.386294    8    1.791759    7.601402    7.601402    ...    0    0    0    1    0    0    0    0    1    0
​5    85    9.555064    1.098612    1.386294    1.609438    1.386294    5    1.791759    7.597898    7.598900    ...    0    0    0    1    0    0    0    0    1    0
​6    75    9.218804    1.098612    1.609438    1.609438    1.386294    8    1.791759    7.603399    7.603898    ...    0    0    0    1    0    0    0    0    1    0
​7    0    9.247925    1.098612    1.386294    1.609438    1.386294    7    1.945910    7.587817    7.587817    ...    0    0    0    1    0    0    0    0    1    0
​8    51    8.719481    1.098612    1.609438    1.609438    1.386294    7    1.791759    7.566311    7.576097    ...    0    0    0    1    1    0    0    0    0    0
​9    50    8.912069    1.098612    1.609438    1.609438    1.386294    5    1.945910    7.570443    7.576097    ...    0    0    0    1    0    0    0    0    1    0
​10    70    9.323758    1.098612    1.609438    1.609438    1.386294    5    1.791759    7.583756    7.583756    ...    0    0    0    1    0    0    0    0    1    0
​11    85    9.386392    1.098612    1.386294    1.609438    1.386294    9    1.791759    7.603898    7.604396    ...    0    1    0    0    0    0    0    0    0    1
​12    0    9.470317    1.098612    1.098612    1.609438    1.386294    5    1.945910    7.582229    7.582229    ...    0    0    0    1    0    0    0    0    1    0
​13    91    9.273597    1.098612    1.386294    1.609438    1.386294    7    1.791759    7.604396    7.604894    ...    0    1    0    0    0    0    0    0    0    1
​14    0    9.298443    1.098612    1.386294    1.609438    1.386294    6    1.791759    7.581210    7.581210    ...    0    0    0    1    0    0    0    0    1    0
​15    51    8.719481    1.098612    1.609438    1.609438    1.386294    7    2.197225    7.565275    7.601902    ...    0    0    0    1    0    0    0    0    1    0
​16    0    9.327412    1.098612    1.386294    1.609438    1.386294    6    2.079442    7.586296    7.586296    ...    0    0    0    1    0    0    0    0    1    0
​17    72    9.286560    1.098612    1.609438    1.609438    1.386294    4    1.791759    7.584773    7.584773    ...    0    0    0    1    0    0    0    0    1    0
​18    66    9.524859    1.098612    1.609438    1.609438    1.386294    5    1.791759    7.603399    7.603399    ...    0    0    0    1    0    0    0    0    1    0
​19    70    8.930759    1.098612    1.609438    1.609438    1.386294    5    1.945910    7.580189    7.583756    ...    0    0    0    0    1    0    0    0    0    0
​20    101    9.562123    1.098612    1.386294    1.609438    1.386294    8    1.791759    7.603898    7.604396    ...    0    1    0    0    0    0    0    0    0    1
​20 rows × 290 columns
​
​問題２：ホールド・アウト法によるデータの分割をしてください。
​条件：テストデータの割合は3割、random_stateは0、変数は「X_train, X_test, y_train, y_test」を使用
​
​In [4]:
​
​from sklearn.model_selection import train_test_split
​X_train,X_test,y_train,y_test=train_test_split(data,y,test_size=0.3,random_state=0)
​問題３：線形回帰モデルを作成してください。
​モジュールのインポート
​インスタンスの生成
​モデルへのfit
​scoreの表示
​In [5]:
​
​from sklearn.linear_model import LinearRegression
​In [6]:
​
​lr=LinearRegression()
​In [7]:
​
​lr.fit(X_train,y_train)
​Out[7]:
​LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
​In [8]:
​
​lr.score(X_test,y_test)
​Out[8]:
​0.69241069574332148
​問題４：Ridge回帰モデルを作成してください。
​モジュールのインポート
​インスタンスの生成（引数：alpha=10）
​モデルへのfit
​テストデータでのscoreの表示
​In [9]:
​
​from sklearn.linear_model import Ridge
​In [12]:
​
​model_ridge=Ridge(alpha=10)
​In [13]:
​
​model_ridge.fit(X_train,y_train)
​
​Out[13]:
​Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None,
​normalize=False, random_state=None, solver='auto', tol=0.001)
​In [15]:
​
​model_ridge.score(X_test,y_test)
​Out[15]:
​0.80901413643448761
​問題５：LASSOモデルを作成してください。
​モジュールのインポート
​インスタンスの生成（引数：alpha=200）
​モデルへのfit
​テストデータでのscoreの表示
​In [16]:
​
​from sklearn.linear_model import Lasso
​In [17]:
​
​model_lasso=Lasso(alpha=200)
​In [18]:
​
​model_lasso.fit(X_train,y_train)
​Out[18]:
​Lasso(alpha=200, copy_X=True, fit_intercept=True, max_iter=1000,
​normalize=False, positive=False, precompute=False, random_state=None,
​selection='cyclic', tol=0.0001, warm_start=False)
​In [19]:
​
​model_lasso.score(X_test,y_test)
​Out[19]:
​0.80810559821732075
​問題６：Elastic Netモデルを作成してください。
​モジュールのインポート
​インスタンスの生成（引数：alpha=0.1, l1_ratio=0.9）
​モデルへのfit
​scoreの表示
​In [20]:
​
​from sklearn.linear_model import ElasticNet
​In [21]:
​
​model_en=ElasticNet(alpha=0.1,l1_ratio=0.9)
​In [22]:
​
​model_en.fit(X_train,y_train)
​/Users/yoshizawarikuto/.pyenv/versions/anaconda3-5.0.0/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
​ConvergenceWarning)
​Out[22]:
​ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.9,
​max_iter=1000, normalize=False, positive=False, precompute=False,
​random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
​In [23]:
​
​model_en.score(X_test,y_test)
​Out[23]:
​0.80902849085614492
