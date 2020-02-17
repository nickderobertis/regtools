from typing import Sequence, Optional

from regtools.ext_statsmodels import summary_col
import pandas as pd

from regtools.fe.output import add_fixed_effects_rows
from regtools.fe.tools import extract_all_dummy_cols_from_dummy_cols_dict_list, \
    extract_all_fe_names_from_dummy_cols_dict_list
from regtools.controls import suppress_controls_in_summary_df
from regtools.cluster.output import add_cluster_rows
from regtools.summarize.split import get_var_df_and_non_var_df
from regtools.summarize.yesno import col_boolean_dict_from_list_of_lists_of_columns
from regtools.summarize.tstat import replace_stderr_with_t_stat_in_summary_df


def produce_summary(reg_list, stderr: bool = False, t_stats: bool = False, float_format: str = '%0.1f',
                    regressor_order: Sequence[str] = tuple(),
                    suppress_other_regressors: bool = False, model_names: Optional[Sequence[str]] = None):
    """
    Produce a summary from a list of regression results

    :param reg_list: list of statsmodels regression results
    :param stderr: set to True to keep rows for standard errors below coefficient estimates
    :param t_stats: set to True to keep rows for standard errors below coefficient estimates and convert them to t-stats
    :param float_format: format string for how to format results in summary
    :param regressor_order: sequence of column names to put first in the regression results
    :param suppress_other_regressors: True for when using regressor_order to suppress coefficients
        that are not in regressor_order into "Controls: Yes". False to keep coefficients
    :param model_names: If a collection is passed, will be used as column names in summary table.
    :return: a regression summary
    :rtype:
    """

    if isinstance(regressor_order, tuple):
        regressor_order = list(regressor_order)

    _check_produce_summary_inputs(
        regressor_order,
        suppress_other_regressors,
        model_names,
        len(reg_list),
        stderr,
        t_stats
    )

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
    split_rows = [var for var in info_dict]

    # Convert stderrs to t-stats if necessary
    if t_stats:
        summ.tables[0] = replace_stderr_with_t_stat_in_summary_df(summ.tables[0], split_rows, reg_list)

    # Handle fe - remove individual fe cols and replace with e.g. Industry Fixed Effects No, Yes, Yes
    dummy_col_dicts = [result.dummy_cols_dict for result in reg_list]
    if any([dummy_col_dict is not None for dummy_col_dict in dummy_col_dicts]): #if fixed effects
        _remove_fe_cols_replace_with_fixed_effect_yes_no_lines(summ, dummy_col_dicts, split_rows)

    # Handle dropping of unimportant coefficients and replacing with Controls: Yes or No
    if suppress_other_regressors:
        summ.tables[0] = suppress_controls_in_summary_df(summ.tables[0], regressor_order, dummy_col_dicts,
                                                         info_dict)

    # Add Yes and No for each cluster variable
    _add_cluster_yes_no_lines(summ, reg_list, split_rows)

    if not stderr and not t_stats:
        summ.tables[0].drop('', axis=0, inplace=True)  # drops the rows containing standard errors

    # Change const to Intercept in output
    summ.tables[0].index = [col if col != 'const' else 'Intercept' for col in summ.tables[0].index]

    if model_names:
        summ.tables[0].columns = model_names

    return summ


def _add_cluster_yes_no_lines(summ, reg_list, split_rows):
    cluster_list_of_lists = _get_cluster_list_of_lists(reg_list)
    if not any([cluster is not None for cluster in cluster_list_of_lists]):
        return

    cluster_col_boolean_dict = col_boolean_dict_from_list_of_lists_of_columns(cluster_list_of_lists)
    var_df, split_df = get_var_df_and_non_var_df(summ.tables[0], split_rows=split_rows)

    var_df = add_cluster_rows(var_df, cluster_col_boolean_dict)

    # Recombine with n, R^2, etc. and
    summ.tables[0] = pd.concat([var_df, split_df], axis=0)

def _get_cluster_list_of_lists(reg_list):
    cluster_list_of_lists = []
    for result in reg_list:
        if hasattr(result, 'cluster_variables'):
            cluster_list_of_lists.append(result.cluster_variables)
        else:
            cluster_list_of_lists.append(None)
    return cluster_list_of_lists


def _remove_fe_cols_replace_with_fixed_effect_yes_no_lines(summ, dummy_col_dicts, split_rows):
    """
    Note: inplace
    """
    # split into dataframe of variables and dataframe of N, R^2, etc.
    var_df, split_df = get_var_df_and_non_var_df(summ.tables[0], split_rows=split_rows)
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


def _check_produce_summary_inputs(regressor_order, supress_other_regressors, model_names, num_models,
                                  stderr: bool, t_stats: bool):
    if (regressor_order == []) & (supress_other_regressors):
        raise ValueError('must pass regressors to regressor_order to suppress other regressors')

    if model_names and (len(model_names) != num_models):
        raise ValueError(f'must pass model_names of equal length to num models. Have {len(model_names)} names and {num_models} models.')

    if stderr and t_stats:
        raise ValueError(f'cannot pass both stderr and t stats, pick one of the two or neither')

def _result_has_adjusted_r2(result):
    return hasattr(result, 'rsquared_adj')

def _result_has_pseudo_r2(result):
    return hasattr(result, 'prsquared')