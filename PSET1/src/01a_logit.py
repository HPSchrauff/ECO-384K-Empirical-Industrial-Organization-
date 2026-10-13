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
import pyblp


MAIN = Path(__file__).resolve().parents[1]

DATA = MAIN / "data"
CODE = MAIN / "src"

for directory in (DATA, CODE):
    directory.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA / 'io_pset_1_data.xlsx')

######################################
#### Question 1 ####
######################################

# renaming columns that have reserved names in pyblp
df = df.rename(columns={'Market_ID': 'market_ids', 'Plan_ID': 'product_ids', 'Share': 'shares', 'Premium': 'prices'})

###################
#### 1.a ####
###################
df_ols = df
# Outside Share
df_ols['TotalShare'] = df.groupby(["market_ids"])['shares'].transform("sum")
df_ols['Outside_Share'] = 1 - df_ols['TotalShare']

# Log of Shares
df_ols['log_os'] = np.log(df_ols['Outside_Share'])
df_ols['log_s'] = np.log(df_ols['shares'])

# Difference 
df_ols['ShareDiff'] = df_ols['log_s'] - df_ols['log_os']

# Naive OLS
y = df_ols['ShareDiff']
x = df_ols[['HMO', 'Network', 'Satisfaction', 'prices']]
X = sm.add_constant(x)

naive_ols = sm.OLS(y, X).fit()
print(naive_ols.summary())

# Let's now get the naive OLS estimate using PyBLP
# OLS treats prices as exogenous whereas PyBLP treats prices as endogenous. We will use the reserved name demand_instruments0 to instrument for prices with prices. This will let us match the OLS results above.  
df['demand_instruments0'] = df['prices']

ols_problem = pyblp.Problem(
    pyblp.Formulation(
        '1 + HMO + Network + Satisfaction + prices'
    ),
    df
)

ols_results = ols_problem.solve(method='1s')
