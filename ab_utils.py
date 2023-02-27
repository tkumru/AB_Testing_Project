import logging
import sys

import pandas as pd

import statsmodels.stats.api as sms

from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu 
    
def setup_logger(name=__name__):
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler(sys.stdout)
    
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    logger.addHandler(c_handler)

    return logger
    
def get_confidence_intervals(df: pd.DataFrame, 
                             col: str) -> tuple:
    """
    Function generate confidence intervals for dataframe column. It generaly 
    set lowest and highest value of dataframe column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str
        Column name.

    Returns
    -------
    tuple
        (lowest, highest).

    """
    return sms.DescrStatsW(df[col]).tconfint_mean()

def check_normality(query, plot: bool) -> bool:
    """
    Function calculate test_stat and p_value. It decides "Is there normal 
    distribution in dataframe values?" for p_value.
    
    Normality Hypothesis = There is normal distribution hypothesis.
    
    If p_value < 0.05, have to say there is no normal distribution exactly but
    the other condition, can to say there can be normal distribution.

    Parameters
    ----------
    query : OPTIONAL
        Dataframe output for aggregation.
    plot : bool
        Drawing plot selection.

    Returns
    -------
    bool
        Is there normal distribution.

    """
    logg = setup_logger()
    
    try:
        test_stat, p_value = shapiro(query)
        
        logg.info(f"Test Stat = {test_stat: .4f}\tp_value = {p_value: .4f}")
        
        if plot: query.plot.kde(title="Normal Distribution")
        
        if not p_value < 0.05:
            logg.debug("Normality Hypothesis cannot be rejected.")
            return True 
        else: 
            logg.debug("Normality Hypothesis was rejected.")
            return False
    except Exception as e:
        logg.error(f"Normality cannot checked!\n{e}")
        return False
    
def check_homogeneity(query1, query2) -> bool:
    """
    Function calculate test_stat and p_value. It decides "Is there Variance 
    Homogeneity in dataframe values?" for p_value.
    
    Variance Homogeneity = Variance is homogeneity.
    
    If p_value < 0.05, have to say there is no variance homogeneity exactly but
    the other condition, can to say there can be variance homogeneity.

    Parameters
    ----------
    query1 : OPTIONAL
        Dataframe output for aggregation.
    query2 : OPTIONAL
        Dataframe output for aggregation.

    Returns
    -------
    bool
        Is there variance homogeneity.

    """
    logg = setup_logger()
    
    try:      
        test_stat, p_value = levene(query1, query2)
        
        logg.info(f"Test Stat = {test_stat: .4f}\tp_value = {p_value: .4f}")
        
        if not p_value < 0.05:
            logg.debug("Variance Homogeneity cannot be rejected.")
            return True 
        else: 
            logg.debug("Variance Homogeneity was rejected.")
            return False
    except Exception as e:
        logg.error(f"Variance Homogeneity cannot checked!\n{e}")
        return False
    
def apply_ab_test(df: pd.DataFrame, 
                      col_name: str, 
                      col_value1: str, 
                      col_value2: str, 
                      col_main: str,
                      plot: bool=False) -> bool:
    """
    Function calculates the main hypothesis answer. 

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col_name : str
        Comparing column name.
    col_value1 : str
        First comparing column value.
    col_value2 : str
        Second comparing column value.
    col_main : str
        Main column name.
    plot : bool, optional
        Drawing plot selection. The default is False.

    Returns
    -------
    bool
        Hypothesis questions answer.

    """
    logg = setup_logger()
    
    ###############################
    # Dataframe Operations
    ###############################
    
    query1 = df.loc[df[col_name] == col_value1, col_main].dropna()
    query2 = df.loc[df[col_name] == col_value2, col_main].dropna()
    
    ###############################
    # Checking Normality Hypothesis
    ###############################
    
    logg.info(f"Checking normal distribution for {col_name} == {col_value1}:")
    normality1 = check_normality(query1, plot)
    
    logg.info(f"Checking normal distribution for {col_name} == {col_value2}:")
    normality2 = check_normality(query2, plot)
    
    normality = normality1 and normality2
    
    ###############################
    # Checking Variance Homogeneity
    ###############################
    
    logg.info("Checking variance homogeneity:")
    homogeneity = check_homogeneity(query1, query2)
    
    ###############################
    # Apply to Param./NonParam Test
    ###############################
    
    result = apply_parametric(query1, query2, homogeneity) if normality else apply_nonparametric(query1, query2)
    
    return result

def apply_parametric(query1, query2,
                      homogen: bool) -> bool:
    """
    Function calculate test_stat and p_value. It decides "What is main 
    hypothesis trueness?" for p_value.
    
    Main Hypothesis = There is differences between each of them.
    
    If p_value < 0.05, have to say there is no differences exactly but
    the other condition, can to say there can be differences.

    Parameters
    ----------
    query1 : OPTIONAL
        Dataframe output for aggregation.
    query2 : OPTIONAL
        Dataframe output for aggregation.
    homogen : bool
        Variance homogeneity result.

    Returns
    -------
    bool
        Is there differences between each of them.

    """
    logg = setup_logger()
    logg.info("Parametric test is starting for passed tests.")
    
    try:        
        test_stat, p_value = ttest_ind(query1, query2, equal_var=homogen)
        
        logg.info(f"Test Stat = {test_stat: .4f}\tp_value = {p_value: .4f}")
        
        if not p_value < 0.05:
            logg.debug("Main Hypothesis cannot be rejected.")
            return True 
        else: 
            logg.debug("Main Hypothesis was rejected.")
            return False
    except Exception as e:
        logg.error(f"Parametric Test cannot checked!\n{e}")
        return False
    
def apply_nonparametric(query1, query2) -> bool:
    """
    Function calculate test_stat and p_value. It decides "What is main 
    hypothesis trueness?" for p_value.
    
    Main Hypothesis = There is differences between each of them.
    
    If p_value < 0.05, have to say there is no differences exactly but
    the other condition, can to say there can be differences.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col_name : str
        Comparing column name.
    col_value1 : str
        First comparing column's value.
    col_value2 : str
        Second comparing column's value.
    col_main : str
        Affected column name.

    Returns
    -------
    bool
        Is there differences between each of them.

    """
    logg = setup_logger()
    logg.info("Non-Parametric test is starting for passed tests.")
    
    try:
        test_stat, p_value = mannwhitneyu(query1, query2)
        
        logg.info(f"Test Stat = {test_stat: .4f}\tp_value = {p_value: .4f}")
        
        if not p_value < 0.05:
            logg.debug("Main Hypothesis cannot be rejected.")
            return True 
        else: 
            logg.debug("Main Hypothesis was rejected.")
            return False
    except Exception as e:
        logg.error(f"Parametric Test cannot checked!\n{e}")
        return False
