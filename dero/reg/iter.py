import itertools

from dero.reg.order import _set_regressor_order
from .chooser import any_reg
from .select import select_models
from .summarize import produce_summary
from .lag import remove_lag_names_from_reg_results


def reg_for_each_combo(df, yvar, xvars, reg_type='reg', **reg_kwargs):
    """
    Takes each possible combination of xvars (starting from each var individually, then each pair
    of vars, etc. all the way up to all xvars), and regresses yvar on each set of xvars. Returns
    a list of fitted regressions.
    """
    reg_list = []
    for i in range(1, len(xvars) + 1):
        for combo in itertools.combinations(xvars, i):
            x = list(combo)
            reg_list.append(any_reg(reg_type, df, yvar, x, **reg_kwargs))

    return reg_list


def reg_for_each_xvar_set(df, yvar, xvars_list, reg_type='reg', **reg_kwargs):
    """
    Runs regressions on the same y variable for each set of x variables passed. xvars_list
    should be a list of lists, where each individual list is one set of x variables for one model.
    Returns a list of fitted regressions.

    If fe is passed, should either pass a string to use fe in all models, or a list of strings or
    None of same length as num models
    """
    fe, interaction_tuples = _pop_and_convert_kwargs_which_are_repeated_across_models(reg_kwargs, len(xvars_list))
    return [any_reg(reg_type, df, yvar, x, fe=fe[i], interaction_tuples=interaction_tuples[i], **reg_kwargs) for i, x in enumerate(xvars_list)]


def reg_for_each_xvar_set_and_produce_summary(df, yvar, xvars_list, robust=True,
                                              cluster=False, stderr=False, fe=None, float_format='%0.2f',
                                              suppress_other_regressors=False,
                                              regressor_order=[], **other_reg_kwargs):
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
    reg_list = reg_for_each_xvar_set(df, yvar, xvars_list, robust=robust, cluster=cluster, fe=fe, **other_reg_kwargs)
    regressor_order = _set_regressor_order(regressor_order, other_reg_kwargs)
    summ = produce_summary(reg_list, stderr=stderr, float_format=float_format,
                           regressor_order=regressor_order, suppress_other_regressors=suppress_other_regressors)
    return reg_list, summ

def reg_for_each_yvar_and_produce_summary(df, yvars, xvars, main_ivs, reg_type='reg',
                                          stderr=False, float_format='%0.2f',
                                          **reg_kwargs):
    reg_list = reg_for_each_yvar(df, yvars, xvars, reg_type=reg_type, **reg_kwargs)
    regressor_order = _set_regressor_order(main_ivs, reg_kwargs)
    summ = produce_summary(reg_list, stderr=stderr, float_format=float_format,
                           regressor_order=regressor_order, suppress_other_regressors=True)

    # Transposed is a better layout since all controls are suppressed and there may be many models
    summ.tables[0] = summ.tables[0].T

    return reg_list, summ



def reg_for_each_yvar(df, yvars, xvars, reg_type='reg', **reg_kwargs):
    return [any_reg(reg_type, df, yvar, xvars, **reg_kwargs) for yvar in yvars]



def reg_for_each_combo_select_and_produce_summary(df, yvar, xvars, robust=True, cluster=False,
                                                  keepnum=5, stderr=False, float_format='%0.1f',
                                                  regressor_order=[],
                                                  **other_reg_kwargs):
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
    reg_list = reg_for_each_combo(df, yvar, xvars, robust=robust, cluster=cluster, **other_reg_kwargs)
    regressor_order = _set_regressor_order(regressor_order, other_reg_kwargs)
    outlist = select_models(reg_list, keepnum, xvars)
    summ = produce_summary(outlist, stderr=stderr, float_format=float_format, regressor_order=regressor_order)
    return outlist, summ

def reg_for_each_lag_and_produce_summary(df, yvar, xvars, main_ivs, lag_tuple=(1,2,3,4), consolidate_lags=True,
                                         reg_type='reg',
                                          stderr=False, float_format='%0.2f',
                                         suppress_other_regressors=False,
                                          **reg_kwargs):
    reg_list = reg_for_each_lag(df, yvar, xvars, lag_tuple=lag_tuple, reg_type=reg_type, **reg_kwargs)

    # If we are consolidating lag coefficients, remove the lag names from the coefficient names
    if consolidate_lags:
        reg_list = remove_lag_names_from_reg_results(reg_list, lag_tuple)
    else:
        # Not consolidating lag coefficients. Therefore regressor order needs to be modified to include
        # renamed lag coefficients.
        # Add to kwargs so regressor order can be set properly
        reg_kwargs.update({'lag_tuple': lag_tuple})

    regressor_order = _set_regressor_order(main_ivs, reg_kwargs)

    model_names = [f'({lag})' for lag in lag_tuple]

    summ = produce_summary(reg_list, stderr=stderr, float_format=float_format,
                           regressor_order=regressor_order, suppress_other_regressors=suppress_other_regressors,
                           model_names=model_names)

    return reg_list, summ

def reg_for_each_lag(df, yvar, xvars, lag_tuple=(1,2,3,4), reg_type='reg', **reg_kwargs):
    return [any_reg(reg_type, df, yvar, xvars, num_lags=lag, **reg_kwargs) for lag in lag_tuple]

def _pop_and_convert_kwargs_which_are_repeated_across_models(reg_kwargs, num_models):
    if 'fe' in reg_kwargs:
        fe = reg_kwargs.pop('fe')
    else:
        fe = None

    if 'interaction_tuples' in reg_kwargs:
        interaction_tuples = reg_kwargs.pop('interaction_tuples')
    else:
        interaction_tuples = []

    fe = _set_fe(fe, num_models)
    interaction_tuples = _set_interaction_tuples(interaction_tuples, num_models)

    return fe, interaction_tuples


def _set_fe(fe, num_models):
    return _set_for_multiple_models(fe, num_models, param_name='fixed effects')

def _set_interaction_tuples(interaction_tuples, num_models):
    return _set_for_multiple_models(interaction_tuples, num_models, param_name='interaction tuples')

def _set_for_multiple_models(param, num_models, param_name='fixed effects'):
    # Here we are being passed a list of strings or None matching the size of models.
    # This is the correct format so just output
    if (isinstance(param, list)) and (len(param) == num_models):
        out_param = param

    # Here we are being passed a single item or a list with a single item
    # Need to expand to cover all models
    else:
        if (not isinstance(param, list)):
            param = [param]
        if len(param) > 1:
            raise ValueError(
                f'Incorrect shape of items for {param_name} passed. Got {len(param)} items, was expecting {num_models}')
        out_param = [param[0]] * num_models

    # Final input checks
    assert isinstance(out_param, list)

    return out_param
