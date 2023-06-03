import math
import random
import numpy as np
import pandas as pd
from scipy.stats import gamma
from Helpers import helpers

def spherepicking(n):
    while True:           #to get rid off [0,0,0,0] case
        l = [random.gauss(0, 1) for i in range(n)]
        sumsq = sum([x * x for x in l])
        if sumsq > 0:
            break
    norm = 1.0 / math.sqrt(sumsq)
    pt = [x * norm for x in l]
    return pt

def ct(r, a):
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si*co*r

def generate_nd_laplace_noise(x, epsilon):
    n = len(x)
    sphere_noise = spherepicking(n)
    r = gamma.rvs(n, scale=1/epsilon)
    u = ct(r, sphere_noise)
    z = x + u
    return z

def generate_nd_laplace_noise_for_dataset(dataframe: pd.DataFrame, epsilon: float):
    Z = []
    for index, row in dataframe.iterrows():
        perturbed_row = generate_nd_laplace_noise(row, epsilon)
        Z.append(perturbed_row)
    return pd.DataFrame(Z, columns=dataframe.columns)

def generate_truncated_nd_laplace_noise_for_dataset(X: pd.DataFrame, epsilon):
    Z_pd = generate_nd_laplace_noise_for_dataset(X, epsilon)
    Z = Z_pd
    return helpers.truncate_n_dimensional_laplace_noise(Z, X, 10)