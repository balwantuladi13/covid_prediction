import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np
from copy import deepcopy
from numpy import inf
from math import exp, gamma
from datetime import timedelta
from sklearn.metrics import r2_score
import matplotlib.patheffects as PathEffects
from scipy.special import softmax
# import warnings
# import os
# import math
from scipy.stats import pearsonr, spearmanr

def grouped_country():

    # df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
    df = pd.read_csv("owid-covid-data.csv")
    df['date'] = pd.to_datetime(df.date)

    df.replace([np.inf, -np.inf, np.nan, ''], 0, inplace=True)
    grouped_country=df.groupby(["location","date"]).agg({"total_cases":'sum',"new_cases":'sum',"total_deaths":'sum', "new_deaths":'sum', "total_tests":'sum', "new_tests":'sum' }).to_dict()
    return grouped_country
