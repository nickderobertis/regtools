from .reg import reg
from ..ext_pandas.filldata import add_missing_group_rows, drop_missing_group_rows
from dero.reg.lag.create import _is_special_lag_keyword


def diff_reg(df, yvar, xvars, id_col, date_col, difference_lag=1, diff_cols=None, **reg_kwargs):

    if diff_cols == None:
        # All by default
        diff_cols = [yvar] + xvars

    df = create_differenced_variables(df, diff_cols, id_col=id_col, date_col=date_col, difference_lag=difference_lag)

    # Convert names in lists of variables being passed to reg
    reg_yvar, reg_xvars = _convert_variable_names(yvar, xvars, diff_cols)
    this_reg_kwargs = reg_kwargs.copy()
    if 'interaction_tuples' in reg_kwargs:
        this_reg_kwargs['interaction_tuples'] = _convert_interaction_tuples(reg_kwargs['interaction_tuples'], diff_cols)
    if 'lag_variables' in reg_kwargs:
        this_reg_kwargs['lag_variables'] = _convert_list_of_variables_to_difference_names(reg_kwargs['lag_variables'], diff_cols)


    result = reg(df, reg_yvar, reg_xvars, **this_reg_kwargs)

    differenced_names = [col + ' Change' for col in diff_cols]
    df.drop(differenced_names, axis=1, inplace=True)

    return result



def create_differenced_variables(df, diff_cols, id_col='TICKER', date_col='Date', difference_lag=1):
    """
    Note: partially inplace
    """
    df.sort_values([id_col, date_col], inplace=True)
    df = add_missing_group_rows(df, [id_col, date_col])

    for col in diff_cols:
        _create_differenced_variable(df, col, id_col=id_col, difference_lag=difference_lag)

    df = drop_missing_group_rows(df, [id_col, date_col])

    return df


def _create_differenced_variable(df, col, id_col='TICKER', difference_lag=1, keep_lag=False):
    """
    Note: inplace
    """
    df[col + '_lag'] = df.groupby(id_col)[col].shift(difference_lag)
    df[col + ' Change'] = df[col] - df[col + '_lag']

    if not keep_lag:
        df.drop(col + '_lag', axis=1, inplace=True)

def _convert_variable_names(yvar, xvars, diff_cols):
    if yvar in diff_cols:
        yvar = yvar + ' Change'

    out_xvars = _convert_list_of_variables_to_difference_names(xvars, diff_cols)

    return yvar, out_xvars

def _convert_list_of_variables_to_difference_names(varlist, diff_cols):

    # if 'all' or 'xvars' is passed, no conversion needed
    if _is_special_lag_keyword(varlist):
        return varlist

    out_vars = []
    for var in varlist:
        if var in diff_cols:
            out_vars.append(var + ' Change')
        else:
            out_vars.append(var)
    return out_vars

def _convert_interaction_tuples(interaction_tuples, diff_cols):
    out_tuples = []
    for tup in interaction_tuples:
        out_tuples.append(tuple([var + ' Change' if var in diff_cols else var for var in tup]))

    return out_tuples


def _is_diff_reg_str(reg_str):
    return reg_str in ('diff', 'difference', 'diff_reg', 'diff reg', 'difference reg', 'difference regression')