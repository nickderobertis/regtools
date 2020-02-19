import warnings

from regtools.differenced import _is_diff_reg_str
from regtools.quantile import _is_quantile_reg_str
from regtools.models import _is_regular_reg_str, _is_ols_str
from regtools.linmodels.bindings.modelstr import _is_linearmodels_str
from regtools.linmodels.reg import linear_reg
from .differenced import diff_reg
from .quantile import quantile_reg
from .reg import reg
from regtools.lag.create import _set_lag_variables


def any_reg(reg_type, *reg_args, **reg_kwargs):
    """
    Runs any regression.

    :param reg_type: 'diff' for difference regression, 'ols' for OLS, 'probit' for Probit, 'logit' for Logit,
        'quantile' for Quantile, or 'fmb' for Fama-MacBeth
    :param reg_args:
    :param reg_kwargs:
    :return:
    """

    reg_args, reg_kwargs = _validate_inputs(reg_type, *reg_args, **reg_kwargs)

    reg_type = reg_type.lower()

    if _is_diff_reg_str(reg_type):
        temp_kwargs = reg_kwargs.copy()
        if 'diff_cols' in temp_kwargs:
            diff_cols = temp_kwargs.pop('diff_cols')
        else:
            diff_cols = None
        id_col = temp_kwargs.pop('id_col')
        date_col = temp_kwargs.pop('date_col')
        return diff_reg(*reg_args, id_col, date_col, diff_cols=diff_cols, **temp_kwargs)

    if _is_regular_reg_str(reg_type):
        if _is_ols_str(reg_type) and 'fe' in reg_kwargs and reg_kwargs['fe'] is not None:
            # More efficient to use linear models for fe as can difference data rather than using dummy variables
            return linear_reg(*reg_args, **reg_kwargs)
        if 'entity_var' in reg_kwargs:
            reg_kwargs.pop('entity_var')
        if 'time_var' in reg_kwargs:
            reg_kwargs.pop('time_var')
        return reg(*reg_args, **reg_kwargs)

    if _is_quantile_reg_str(reg_type):
        if 'reg_type' in reg_kwargs:
            reg_kwargs.pop('reg_type')
        return quantile_reg(*reg_args, **reg_kwargs)

    if _is_linearmodels_str(reg_type):
        return linear_reg(*reg_args, **reg_kwargs)

    raise ValueError(f'Must pass valid reg type. Got {reg_type}')


def _validate_inputs(reg_type: str, *args, **kwargs):
    yvar = args[1]
    xvars = args[2]

    # If an x variable is actually going to be lagged, don't need to exclude it if it matches the y variable
    if 'num_lags' in kwargs and kwargs['num_lags'] > 0:
        lagvars = _set_lag_variables(kwargs['lag_variables'] if 'lag_variables' in kwargs else [], yvar, xvars)
        if yvar in lagvars:
            # if y variable lagged, then remove any xvars that are also lagged
            # only consider those which are in lag variables
            examine_xvars = [var for var in lagvars if var != yvar]
        else:
            # y variable not lagged. shouldn't remove any xvars which are in lag variables
            # only consider those not in lag variables
            examine_xvars = [xvar for xvar in xvars if xvar not in lagvars]
    else:
        examine_xvars = xvars

    # Exclude x variable when matches y variable
    if yvar in examine_xvars:
        warnings.warn(f'{yvar} is both Y variable and passed in X variables. Removing from X for this model.', UserWarning)
        new_xvars = xvars.copy()
        new_xvars.remove(yvar)
        args = args[:2] + (new_xvars,) + args[3:]

    kwargs.update(dict(reg_type=reg_type))

    return args, kwargs
