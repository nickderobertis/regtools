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

    If fe is passed, should either pass a string to use fe in all models, or a list of strings or
    None of same length as num models
    """
    if 'fe' in reg_kwargs:
        fe = reg_kwargs.pop('fe')
    else:
        fe = None

    fe = _set_fe(fe, len(xvars_list))
    return [reg(df, yvar, x, fe=fe[i], **reg_kwargs) for i, x in enumerate(xvars_list)]


def reg_for_each_xvar_set_and_produce_summary(df, yvar, xvars_list, robust=True,
                                              cluster=False, stderr=False, fe=None, float_format='%0.2f',
                                              regressor_order=[]):
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
    fe:    If fe is passed, should either pass a string to use fe in all models, or a list of strings or
    None of same length as num models

    Note: only specify at most one of robust and cluster.
    """
    reg_list = reg_for_each_xvar_set(df, yvar, xvars_list, robust=robust, cluster=cluster, fe=fe)
    summ = produce_summary(reg_list, stderr=stderr, float_format=float_format, regressor_order=regressor_order)
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

def _set_fe(fe, num_models):
    # Here we are being passed a list of strings or None matching the size of models.
    # This is the correct format so just output
    if (isinstance(fe, list)) and (len(fe) == num_models):
        out_fe = fe

    # Here we are being passed a single item or a list with a single item
    # Need to expand to cover all models
    else:
        if (not isinstance(fe, list)):
            fe = [fe]
        if len(fe) > 1:
            raise ValueError(
                f'Incorrect shape of items for fixed effects passed. Got {len(fe)} items, was expecting {num_models}')
        out_fe = [fe[0]] * num_models

    # Final input checks
    assert isinstance(out_fe, list)
    assert all([(isinstance(b, (type(None), str))) for b in out_fe])

    return out_fe