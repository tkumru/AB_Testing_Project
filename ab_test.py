import pandas as pd

from ab_utils import get_confidence_intervals, apply_ab_test
    
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
Get Confidence Intervals
"""

get_confidence_intervals(df, "Purchase")

"""
Define The Hypothesis

H0 : M1 = M2
H1 : M1!= M2 

"""

control["Purchase"].mean()
test["Purchase"].mean()

"""
Get Result
"""

apply_ab_test(df, "group", "control", "test", "Purchase", 1)

# There is no differences between test group and control group. 
# Company can return to back or company should make more efficient changes.
