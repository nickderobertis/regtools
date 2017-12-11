import itertools

from .reg import reg
from .summarize import produce_summary
from .select import select_models


def reg_for_each_combo(df, yvar, xvars, **reg_kwargs):
    """
    Takes each possible combination of xvars (starting from each var individually, then each pair
    of vars, etc. all the way up to all xvars), and regresses yvar on each set of xvars. Returns
    a list of fitted regressions.
    """
    reg_list = []
    for i in range(1, len(xvars) + 1):
        for combo in itertools.combinations(xvars, i):
            x = list(combo)
            reg_list.append(reg(df, yvar, x, **reg_kwargs))

    return reg_list


def reg_for_each_xvar_set(df, yvar, xvars_list, **reg_kwargs):
    """
    Runs regressions on the same y variable for each set of x variables passed. xvars_list
    should be a list of lists, where each individual list is one set of x variables for one model.
    Returns a list of fitted regressions.
    """
    return [reg(df, yvar, x, **reg_kwargs) for x in xvars_list]


def reg_for_each_xvar_set_and_produce_summary(df, yvar, xvars_list, robust=True,
                                              cluster=False, stderr=False, float_format='%0.1f'):
    """
    Convenience function to run regressions for every set of xvars passed
    and present them in a summary format. Returns a tuple of (reg_list, summary) where reg_list
    is a list of fitted regression models, and summary is a single dataframe of results.

    Required inputs:
    df: pandas dataframe containing regression data
    yvar: str, column name of y variable
    xvars_list: list of lists of strs, each individual list has column names of x variables for that model

    Optional inputs:
    robust: bool, set to True to use heterskedasticity-robust standard errors
    cluster: False or str, set to a column name to calculate standard errors within clusters
             given by unique values of given column name
    stderr: bool, set to True to keep rows for standard errors below coefficient estimates

    Note: only specify at most one of robust and cluster.
    """
    reg_list = reg_for_each_xvar_set(df, yvar, xvars_list, robust=robust, cluster=cluster)
    summ = produce_summary(reg_list, stderr=stderr, float_format=float_format)
    return reg_list, summ


def reg_for_each_combo_select_and_produce_summary(df, yvar, xvars, robust=True, cluster=False,
                                                  keepnum=5, stderr=False, float_format='%0.1f'):
    """
    Convenience function to run regressions for every combination of xvars, select the best models,
    and present them in a summary format. Returns a tuple of (reg_list, summary) where reg_list
    is a list of fitted regression models, and summary is a single dataframe of results

    Required inputs:
    df: pandas dataframe containing regression data
    yvar: str, column name of y variable
    xvars: list of strs, column names of all possible x variables

    Optional inputs:
    robust: bool, set to True to use heterskedasticity-robust standard errors
    cluster: False or str, set to a column name to calculate standard errors within clusters
             given by unique values of given column name
    keepnum: int, number to keep for each amount of x variables. The total number of outputted
             regressions will be roughly keepnum * len(xvars)
    stderr: bool, set to True to keep rows for standard errors below coefficient estimates

    Note: only specify at most one of robust and cluster.

    """
    reg_list = reg_for_each_combo(df, yvar, xvars, robust=robust, cluster=cluster)
    outlist = select_models(reg_list, keepnum, xvars)
    summ = produce_summary(outlist, stderr=stderr, float_format=float_format)
    return outlist, summ