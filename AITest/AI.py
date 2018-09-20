# -*- coding=GBK -*-
'''
进行python_ai这个虚拟环境之后，我们来安装所需要的模块，其中主要有：

numpy：用于科学计算的基本模块
scipy：科学计算工具箱
pandas：数据分析和处理模块
scikit-learn：机器学习经典算法的集成包
nltk：自然语言处理模块
jieba：中文分词模块
jupyter：一个交互式的笔记本，我们的代码的主战场
https://zmister.com/archives/234.html
数据处理（从原始数据进行各种处理）
生成训练集（从预处理好的数据中）
算法的选择、训练和评估
部署和监控
'''
import numpy as np
import pandas as pd
import scipy
import sklearn
import nltk
import jieba
# print(np.__version__)
# print(pd.__version__)
# print(scipy.__version__)
# print(sklearn.__version__)
# print(nltk.__version__)
# print(jieba.__version__)
from sklearn import preprocessing #数据预处理模块
from sklearn.model_selection import train_test_split #引入训练模块
from sklearn import datasets
data = datasets.load_boston()
print(data.keys())
print(data.feature_names)
#print(data.DESCR)
print(data.data.shape)
#波士顿房价预测
#step 1 引入数据集
data_df = pd.DataFrame(data.data,columns=data.feature_names)
data_df['房价值'] = data.target
print(data_df.head(10))
#step 2 分割训练数据和测试数据
x_train,x_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.15)
#选择一个回归算法估计器
from sklearn.linear_model import LinearRegression
lineModel = LinearRegression()
lineModel.fit(x_train,y_train)
lineModel.predict(x_test)
#导入算法估计器
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(max_depth=10,n_estimators=150)
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
for t,p in zip(y_test[:10],y_pred[:10]):
    print("正确值:",t,">>>预测值:",p,"相差值:",t-p)
#平均绝对误差MAE和均方差MSE以及R2分数来对回归模型评估
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("MSE:",mean_squared_error(y_pred,y_test))
print("MAE:",mean_absolute_error(y_pred,y_test))
print("R2:",r2_score(y_pred,y_test))


