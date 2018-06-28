import warnings

from dero.reg.differenced import _is_diff_reg_str
from dero.reg.quantile import _is_quantile_reg_str
from dero.reg.dataprep import _is_normal_reg_str
from dero.reg.linmodels.bindings.modelstr import _is_linearmodels_str
from dero.reg.linmodels.reg import linear_reg
from .differenced import diff_reg
from .quantile import quantile_reg
from .reg import reg
from dero.reg.lag.create import _set_lag_variables


def any_reg(reg_type, *reg_args, **reg_kwargs):

    reg_args, reg_kwargs = _validate_inputs(*reg_args, **reg_kwargs)

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

    if _is_normal_reg_str(reg_type):
        return reg(*reg_args, **reg_kwargs)

    if _is_quantile_reg_str(reg_type):
        return quantile_reg(*reg_args, **reg_kwargs)

    if _is_linearmodels_str(reg_type):
        return linear_reg(*reg_args, **reg_kwargs)

    raise ValueError(f'Must pass valid reg type. Got {reg_type}')


def _validate_inputs(*args, **kwargs):
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

    return args, kwargs
