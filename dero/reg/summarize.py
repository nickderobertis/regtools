from .ext_statsmodels import summary_col
import pandas as pd

from .fe import add_fixed_effects_rows
from .fe.tools import extract_all_dummy_cols_from_dummy_cols_dict_list, \
    extract_all_fe_names_from_dummy_cols_dict_list
from .controls import suppress_controls_in_summary_df



def produce_summary(reg_sets, stderr=False, float_format='%0.1f', regressor_order=[],
                    suppress_other_regressors=False, model_names=None):
    """

    :param reg_sets:
    :type reg_sets:
    :param stderr:
    :type stderr:
    :param float_format:
    :type float_format:
    :param regressor_order:
    :type regressor_order:
    :param suppress_other_regressors: True for when using regressor_order to suppress coefficients
        that are not in regressor_order into "Controls: Yes". False to keep coefficients
    :type suppress_other_regressors: bool
    :param model_names: If a collection is passed, will be used as column names in summary table.
    :type model_names: None, list, tuple
    :return:
    :rtype:
    """

    _check_produce_summary_inputs(regressor_order, suppress_other_regressors, model_names, len(reg_sets))

    # If not fixed effects, dummy_col_dicts will be None
    reg_list, dummy_col_dicts = _extract_result_list_and_dummy_dicts(reg_sets)

    info_dict = {'N': lambda x: "{0:d}".format(int(x.nobs))}

    # Grab proper r-squared. For OLS, it's adjusted r-squared, for probit and logit, it's Pseudo r-squared
    if _result_has_adjusted_r2(reg_list[0]):
        info_dict.update({
            'Adj-R2': lambda x: "{:.2f}".format(x.rsquared_adj)
        })
    elif _result_has_pseudo_r2(reg_list[0]):
        info_dict.update({
            'Pseudo-R2': lambda x: "{:.2f}".format(x.prsquared)
        })

    summ = summary_col(reg_list, stars=True, float_format=float_format,
                       regressor_order=regressor_order,
                       info_dict=info_dict)

    # Handle fe - remove individual fe cols and replace with e.g. Industry Fixed Effects No, Yes, Yes
    if dummy_col_dicts: #if fixed effects
        split_rows = [var for var in info_dict]
        _remove_fe_cols_replace_with_fixed_effect_yes_no_lines(summ, dummy_col_dicts, split_rows)

    # Handle dropping of unimportant coefficients and replacing with Controls: Yes or No
    if suppress_other_regressors:
        summ.tables[0] = suppress_controls_in_summary_df(summ.tables[0], regressor_order, dummy_col_dicts,
                                                         info_dict)

    if not stderr:
        summ.tables[0].drop('', axis=0, inplace=True)  # drops the rows containing standard errors

    # Change const to Intercept in output
    summ.tables[0].index = [col if col != 'const' else 'Intercept' for col in summ.tables[0].index]

    if model_names:
        summ.tables[0].columns = model_names

    return summ

def _remove_fe_cols_replace_with_fixed_effect_yes_no_lines(summ, dummy_col_dicts, split_rows):
    """
    Note: inplace
    """
    # split into dataframe of variables and dataframe of N, R^2, etc.
    var_df, split_df = _get_var_df_and_non_var_df(summ.tables[0], split_rows=split_rows)
    # get name of all fixed effect variables
    all_cols_to_remove = extract_all_dummy_cols_from_dummy_cols_dict_list(dummy_col_dicts)
    # remove fixed effect coefs and stderrs
    var_df = _drop_variables_from_reg_summary_df(var_df, all_cols_to_remove)

    # construct a single dict where the keys are names of fixed effects and the values are lists of booleans for
    # whether the fixed effect was used
    fe_dict = _multiple_model_fe_dict_from_dummy_col_dict_list(dummy_col_dicts)

    # Add yes no row
    var_df = add_fixed_effects_rows(var_df, fe_dict)

    # Recombine with n, R^2, etc. and
    summ.tables[0] = pd.concat([var_df, split_df], axis=0)

def _extract_result_list_and_dummy_dicts(result_sets):
    plain_results = []
    dummy_dicts = []
    for ambiguous_result in result_sets:
        # This is the case where fe has been passed, and we have a dummy_col_dict
        if isinstance(ambiguous_result, tuple):
            plain_results.append(ambiguous_result[0])
            dummy_dicts.append(ambiguous_result[1])
        # No fe passed, just plain result
        else:
            plain_results.append(ambiguous_result)
            dummy_dicts.append(None)  # keep order by appending None

    return plain_results, dummy_dicts


def _multiple_model_fe_dict_from_dummy_col_dict_list(dummy_col_dict_list):
    fixed_effect_rows = extract_all_fe_names_from_dummy_cols_dict_list(dummy_col_dict_list)
    out_dict = {fe_name: [] for fe_name in fixed_effect_rows}
    for dummy_col_dict in dummy_col_dict_list:
        for fe_name in fixed_effect_rows:
            if (not dummy_col_dict) or (fe_name not in dummy_col_dict):
                out_dict[fe_name].append(False)
            else:
                out_dict[fe_name].append(True)

    return out_dict


def _drop_variables_from_reg_summary_df(df, dropvars):
    # Find variables to be kept
    keepvars = [var for var in df.index if var not in dropvars and var != '']

    # Create column identifying row as an estimate or standard error
    df['type'] = ['estimate', 'stderr'] * int(len(df.index) / 2)

    # Create column identifying variable name of row (no spaces)
    df['regressor'] = [i for sublist in [[j] * 2 for j in df.index[0::2]] for i in sublist]

    # Create a column of the original location of the row (will be sorting index, need to get original
    # order back later)
    df['idx'] = [i for i in range(len(df))]

    # Create the multi-index for slicing
    df = df.set_index(['regressor', 'type'])
    df = df.sort_index()

    # Slice on the chosen regressors, reset the index to delete a column later
    df = df.loc[keepvars].reset_index()
    df = df.sort_values('idx')

    # Set value of index back to original - which had blanks for stderrs
    df.loc[df['type'] == 'stderr', 'regressor'] = ''

    # Delete the type column
    df.drop(['type', 'idx'], axis=1, inplace=True)

    # Reindex the dataframe on the regressor
    df = df.set_index(['regressor'])

    # Get rid of name on index
    df.index.name = None

    return df

def _get_var_df_and_non_var_df(df, split_rows=['N', 'R2', 'Adj-R2']):
    """
    Splits rows containing N, R2, Adj-R2 into separate dataframe
    """
    other_data_mask = df.index.isin(split_rows)
    return df.loc[~other_data_mask], df.loc[other_data_mask]

def _check_produce_summary_inputs(regressor_order, supress_other_regressors, model_names, num_models):
    if (regressor_order == []) & (supress_other_regressors):
        raise ValueError('must pass regressors to regressor_order to suppress other regressors')

    if model_names and (len(model_names) != num_models):
        raise ValueError(f'must pass model_names of equal length to num models. Have {len(model_names)} names and {num_models} models.')

def _result_has_adjusted_r2(result):
    return hasattr(result, 'rsquared_adj')

def _result_has_pseudo_r2(result):
    return hasattr(result, 'prsquared')