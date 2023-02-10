import itertools

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportions_ztest

from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
    
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

control = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")

"""
Impression: Advertisement view count
Click: Viewed adv. click count
Purchase: Clicked sold count
Earning: Price
"""

control.describe().T
test.describe().T

control["group"] = "control"
test["group"] = "test"

df = pd.concat([control, test], axis=0, ignore_index=True)

"""
Define The Hypothesis

H0 : M1 = M2
H1 : M1!= M2 

"""

control["Purchase"].mean()
test["Purchase"].mean()

"""
Normality Hypothesis

H0: There is normal distribution.
H1: There is no normal distribution.

p < 0.05 H0 REJECTED 
p > 0.05 H0 CANNOT REJECTED
"""

p_value_control = shapiro(df.loc[df["group"] == "control", "Purchase"])[1]
p_value_test = shapiro(df.loc[df["group"] == "test", "Purchase"])[1]

# Both p_values higher than 0.05

"""
Variance 	Homogeneity 

H0: Variance is homogeneous.
H1: Variance is not homogeneous.

p < 0.05 H0 REJECTED 
p > 0.05 H0 CANNOT REJECTED
"""

p_value_homo = levene(df.loc[df["group"] == "control", "Purchase"],
                      df.loc[df["group"] == "test", "Purchase"])[1]

# p_value is higher than 0.05

"""
Parametric Test
"""

p_value_param = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                          df.loc[df["group"] == "test", "Purchase"],
                          equal_var=True)[1]

# There is no differences between test group and control group. 
# Company can return to back or company should make more efficient changes.
