from dero.reg.differenced import _is_diff_reg_str
from dero.reg.lag.create import varname_to_lagged_varname


def _set_regressor_order(regressor_order, reg_kwargs):

    # processing needed for lagged regression
    if ('num_lags' in reg_kwargs) or ('lag_tuple' in reg_kwargs):
        regressor_order = convert_regressor_order_for_lags(regressor_order, reg_kwargs)

    # processing needed for difference regression
    if ('reg_type' in reg_kwargs) and (_is_diff_reg_str(reg_kwargs['reg_type'])):
        regressor_order = convert_regressor_order_for_diff(regressor_order, reg_kwargs)

    return regressor_order

def convert_regressor_order_for_diff(regressor_order, reg_kwargs):
    if 'diff_cols' in reg_kwargs:
        cols = reg_kwargs['diff_cols']
    else:
        cols = 'all'

    return _convert_regressor_order_for_diff(regressor_order, cols)

def _convert_regressor_order_for_diff(regressor_order, cols='all'):
    if cols == 'all':
        cols = regressor_order.copy()

    return [col + ' Change' if col in cols else col for col in regressor_order]

def convert_regressor_order_for_lags(regressor_order, reg_kwargs):

    # Set lag variables
    if 'lag_variables' in reg_kwargs:
        lag_cols = reg_kwargs['lag_variables']
    else:
        lag_cols = 'all'

    if lag_cols in ('xvars', 'all'):
        lag_cols = regressor_order.copy()

    # Set number of lags
    if 'num_lags' in reg_kwargs:
        # Here a single number of lags has been passed
        lags = [reg_kwargs['num_lags']]
    elif 'lag_tuple' in reg_kwargs:
        # Here multiple lags are being passed in a tuple
        lags = reg_kwargs['lag_tuple']

    return _convert_regressor_order_for_lags(regressor_order, lag_cols, lags)

def _convert_regressor_order_for_lags(regressor_order, lag_cols, lags):
    out_cols = []
    for col in regressor_order:
        if col in lag_cols:
            [out_cols.append(varname_to_lagged_varname(col, num_lags=num_lags)) for num_lags in lags]
        else:
            out_cols.append(col)

    return out_cols