import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math as m
Utrecht = pd.read_csv('Utrecht.csv', encoding='utf-8')
exp_var = ['bo_yta','ar','tomt','balkong']
#linjära transformer av datan verkar inte påverka prediktionsresultatet!
#Utrecht['ar'] = abs(Utrecht['ar']-1970)

Utrecht_train = Utrecht.iloc[:90,:]
Utrecht_test = Utrecht.iloc[90:,:]


X_test = np.array(Utrecht_test[exp_var])
X_test = np.concatenate((np.ones((10,1),dtype=float),X_test),axis=1)
Y_test = np.array(Utrecht_test['pris']).reshape((10,1))

ones = np.ones((90,1),dtype=float)
Y = np.array(Utrecht_train['pris']).reshape((90,1))
X = np.array(Utrecht_train[exp_var])
X = np.concatenate((ones,X),axis=1)
inv = np.linalg.inv(np.matmul(X.T,X))
mul = np.matmul(inv,X.T)
final = np.matmul(mul,Y)


pred = np.matmul(X_test,final)
errors = Y_test-np.matmul(X_test,final)
#print(pred,Y_test)
print(round(sum(errors**2)[0]))

