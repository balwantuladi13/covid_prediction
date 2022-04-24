import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn import metrics
import datetime
plt.style.use('seaborn')

def get_predictions():

    confirmed_cases = pd.read_csv('confirmed cases.csv')
    cols = confirmed_cases.keys()
    confirmed = confirmed_cases.loc[:, cols[4]:cols[-1]]

    dates = confirmed.keys()
    world_cases = []

    for i in dates:
        confirmed_sum = confirmed[i].sum()
        world_cases.append(confirmed_sum)

    ans = world_cases
    days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    days_in_future = 10
    future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-10]

    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forcast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.2, shuffle=False) 

    kernel = ['poly']
    c = [100]
    gamma = [0.01]
    epsilon = [0.1]

    shrinking = [True,False]
    svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}

    svm = SVR()
    svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
    svm_search.fit(X_train_confirmed, y_train_confirmed)

    svm_confirmed = svm_search.best_estimator_
    svm_pred = svm_confirmed.predict(future_forcast)

    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    model_predictions=pd.DataFrame(zip(future_forcast_dates[-10:], svm_pred[-10:]),
                                columns=["Date    ","      SVM      "])
    #model_predictions.head()
    return model_predictions.to_dict()