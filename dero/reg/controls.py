import pandas as pd

def suppress_controls_in_summary_df(summ_df, regressor_order, dummy_col_dicts, info_dict):
    regressors, fe, info = _each_row_set_to_keep_from_regressor_order_dummy_col_dicts_and_info_dict(
        regressor_order, dummy_col_dicts, info_dict
    )
    variables_df, fe_df, info_df = _split_summ_df_into_variables_fixed_effects_info(summ_df, fe, info)
    regressors_df, controls_df = _split_variable_df_into_regressors_and_controls(variables_df, regressors)
    controls_row = _create_controls_row_as_df(controls_df)
    return _combine_dfs(regressors_df, controls_row, fe_df, info_df)


def _all_rows_to_keep_from_regressor_order_dummy_col_dicts_and_info_dict(regressor_order, dummy_col_dicts, info_dict):
    regressors, fe, info = _each_row_set_to_keep_from_regressor_order_dummy_col_dicts_and_info_dict(
        regressor_order, dummy_col_dicts, info_dict
    )
    return regressors + fe + info


def _each_row_set_to_keep_from_regressor_order_dummy_col_dicts_and_info_dict(regressor_order, dummy_col_dicts,
                                                                             info_dict):
    regressors = regressor_order.copy()
    fe = _extract_all_fe_names_from_dummy_col_dicts(dummy_col_dicts)
    info = [col for col in info_dict]
    return regressors, fe, info


def _extract_all_fe_names_from_dummy_col_dicts(dummy_col_dicts):
    cols = []
    # Loop through models, the dummy variable dictionary for each
    for dummy_dict in dummy_col_dicts:
        # Will come through as None instead of dict if no fixed effects for this model
        if not dummy_dict:
            continue
        for col in dummy_dict:
            full_name = col + ' Fixed Effects'
            if full_name not in cols:
                cols.append(full_name)

    return cols


def _split_summ_df_into_variables_fixed_effects_info(df, fe, info):
    fe_mask = df.index.isin(fe)
    fe_df = df.loc[fe_mask]
    info_mask = df.index.isin(info)
    info_df = df.loc[info_mask]

    variables_mask = ~fe_mask & ~info_mask
    variables_df = df.loc[variables_mask]

    return variables_df, fe_df, info_df


def _split_variable_df_into_regressors_and_controls(variables_df, regressors, stderr=True):
    controls = [row for row in variables_df.index if row not in regressors + ['']]
    all_variables = regressors + controls

    if stderr:
        # Assign index values of name of regressor to stderr rows, that way both coef and stderr
        # are referenced by the same index
        f = lambda idx: sum([[x, x] for x in idx], [])
        variables_df.index = f(all_variables)

    # Select df of just regressors, with modified index
    regressors_mask = variables_df.index.isin(regressors)
    regressors_df = variables_df[regressors_mask]
    controls_df = variables_df[~regressors_mask]

    if stderr:
        # Reset indices so that stderrs have blank indices again
        f = lambda idx: sum([[x, ''] for x in idx], [])
        regressors_df.index = f(regressors)
        controls_df.index = f(controls)

    return regressors_df, controls_df


def _create_controls_row_as_df(controls_df):
    bool_df = pd.DataFrame(controls_df.apply(lambda x: bool(x.any()), axis=0)).T
    yes_no_df = bool_df.applymap(lambda x: 'Yes' if x else 'No')
    yes_no_df.index = ['Controls']

    return yes_no_df


def _combine_dfs(regressors_df, controls_row, fe_df, info_df):
    df_list = [regressors_df, controls_row, fe_df, info_df]
    return pd.concat(df_list, axis=0)

