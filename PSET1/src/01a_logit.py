######################################
#### MODULES AND PATHS ####
######################################

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize 
from scipy import stats
import statsmodels.api as sm 
from pathlib import Path
from math import pi
from math import log
import random


MAIN = Path(__file__).resolve().parents[1]

DATA = MAIN / "data"
CODE = MAIN / "src"

for directory in (DATA, CODE):
    directory.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA / 'io_pset_1_data.xlsx')

######################################
#### Question 1 ####
######################################

###################
#### 1.a ####
###################

# Outside Share
df['TotalShare'] = df.groupby(["Market_ID"])['Share'].transform("sum")
df['Outside_Share'] = 1 - df['TotalShare']

# Log of Shares
df['log_os'] = np.log(df['Outside_Share'])
df['log_s'] = np.log(df['Share'])

# Difference 
df['ShareDiff'] = df['log_s'] - df['log_os']

# Naive OLS
y = df['ShareDiff']
x = df[['HMO', 'Network', 'Satisfaction', 'Premium']]
X = sm.add_constant(x)

naive_ols = sm.OLS(y, X).fit()
print(naive_ols.summary())


