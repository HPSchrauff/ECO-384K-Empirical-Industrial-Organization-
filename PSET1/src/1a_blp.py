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
from statsmodels.sandbox.regression.gmm import IV2SLS

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

# Outside Share
df['TotalShare'] = df.groupby(["market_ids"])['shares'].transform("sum")
df['Outside_Share'] = 1 - df['TotalShare']

# Log of Shares
df['log_os'] = np.log(df['Outside_Share'])
df['log_s'] = np.log(df['shares'])

# Difference 
df['ShareDiff'] = df['log_s'] - df['log_os']

######################################
#### 1. Simulate market shares ####
######################################

# Given the RE utility specification does not imply an analytic form for market shares, we need to simulate market shares. 
# That requires us to simulate, for each market, 500 agents with taste parameters for 3 product characteristics drawn from N(0,1). We do not have a random coefficient on price

# number of markets
M = df['market_ids'].nunique()
# number of draws per market
draws = 500
# number of product characteristics 
K = 3



nu = np.random.standard_normal(size = (M, draws, K))

def calc_shares(delta, sigma, X, markets, nu): 
    """
    
    """
    # We need to loop through each market to match observed with computed shares for a candidate set of parameters
    mu = X @ nu @ sigma 
    delta = delta_old
    f_j = (np.exp(delta + mu))
    
# Let's work with market 1 as an example

market_1 = df[df["market_ids"] == 1]
nu_1 = nu[1]
x = market_1[['HMO', 'Network', 'Satisfaction']]
sigma_1 = np.zeros((3,3))
np.fill_diagonal(sigma_1, 4)
mu = (x @ sigma_1 @ nu_1.T).to_numpy()
f_j1 = np.exp(delta_shaped + mu) / (1 + np.sum(np.exp(delta_shaped + mu), axis=0))
s_j1 = np.average(f_j1, axis=1)


# We need to loop through different delta_j values until we find one such that observable shares are sufficiently close to model-implied shares

# initial guess 
delta_0 = market_1['ShareDiff']
delta = delta_0.to_numpy()
# delta_shaped = np.repeat(delta, draws).reshape(4, draws)
delta_shaped = np.tile(delta, (draws, 1)).T
tol = 1e-6

while error > tol: 
    delta_old = delta 
    delta = delta_old + df['log_s'] - calc_shares(delta_old, sigma, x, price, markets, nu)
    error = np.linalg.norm(delta - delta_old, ord=np.inf)

    return delta









f_j = (np.exp())
    
    
for m in np.unique(df['market_ids']): 



    f_ijt = np.exp(delta_jt + mu_ijt) / (1 + np.sum(np.exp))

