#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:38:09 2018

@author: Ali Zamani----zamaniali1995@gmail.com
"""
# Load libraries
import pandas
import matplotlib.pyplot as plt
from tabulate import tabulate
from pandas.plotting import scatter_matrix
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
plt.style.use('seaborn-white')
#%%
# Load dataset
names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'class']
names1 = ['','sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']
dataSet = pandas.read_csv('Data/iris.csv', names=names,usecols=[0,1,2,3,4])
dataSet.info()
print(dataSet.groupby('class').size())
print ("Data shape:", dataSet.shape)
print (tabulate(dataSet.head(5),headers=names,tablefmt="grid"))
print (tabulate(dataSet.describe(),headers=names1,tablefmt="grid"))
#print(dataset.describe())
#%%
#scatter plot matrix
scatter_matrix(dataSet,grid=True)
#%%
#Linear Regression(petalLength as a fuction of petalWidth)
petalWidth=dataSet.values[:,3].reshape(-1,1)
petalLength=dataSet.values[:,2].reshape(-1,1)
LR= LinearRegression()
LR.fit(petalWidth,petalLength)
print ("Intercept:" , LR.intercept_)
print ("Slope:" , LR.coef_)
petalLength_pred=LR.predict(petalWidth)
LRfig, LRax = plt.subplots( nrows=1, ncols=1 )
LRax.scatter(petalWidth,petalLength)
LRax.plot(petalWidth,petalLength_pred,color='red')
LRfig.savefig('LR.pdf')
#Report the t-value and p-value 
est=smf.ols('petalLength ~ petalWidth', dataSet).fit()
print (est.summary().tables[1])

#%%
sepalWidth=dataSet.values[:,1].reshape(-1,1)
V=petalWidth*sepalWidth
LR_mul= LinearRegression()
LR_mul.fit(V,petalLength)
print ("Intercept:" , LR_mul.intercept_)
print ("Slope:" , LR_mul.coef_)
petalLenMul_pred=LR_mul.predict(V)
LR_mulfig, LR_mulax = plt.subplots( nrows=1, ncols=1 )
LR_mulax.scatter(V,petalLength)
LR_mulax.plot(V,petalLenMul_pred,color='red')
LR_mulfig.savefig('LR_mul.pdf')
#Report the t-value and p-value 
est_mul=smf.ols('petalLength ~ petalWidth+sepalWidth+petalWidth*sepalWidth', dataSet).fit()
print (est_mul.summary().tables[1])

dataSet.corr()  