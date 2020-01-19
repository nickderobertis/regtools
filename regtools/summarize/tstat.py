from typing import Sequence
import pandas as pd
import numpy as np
from regtools.summarize.split import get_var_df_and_non_var_df


def replace_stderr_with_t_stat_in_summary_df(df: pd.DataFrame, split_rows: Sequence[str], reg_list) -> pd.DataFrame:
    var_df, non_var_df = get_var_df_and_non_var_df(df, split_rows=split_rows)

    var_df = replace_stderr_with_t_stat_in_var_df(var_df, reg_list)

    return pd.concat([var_df, non_var_df], axis=0)


def replace_stderr_with_t_stat_in_var_df(df: pd.DataFrame, reg_list) -> pd.DataFrame:
    # Create column identifying row as an estimate or standard error
    df['type'] = ['estimate', 'stderr'] * int(len(df.index) / 2)

    # Create column identifying variable name of row (no spaces)
    df['regressor'] = [i for sublist in [[j] * 2 for j in df.index[0::2]] for i in sublist]

    numeric_cols = [col for col in df.columns if col not in ['regressor', 'type']]

    for regressor in df['regressor'].unique():
        t_values = _extract_regressor_t_df_from_reg_list(reg_list, regressor, numeric_cols)
        t_values = t_values.applymap(convert_to_stderr_format)
        df.loc[
            (df['regressor'] == regressor) &
            (df['type'] == 'stderr'),
            numeric_cols
        ] = t_values

    # Delete the created columns
    df.drop(['type', 'regressor'], axis=1, inplace=True)

    return df


def _extract_regressor_t_df_from_reg_list(reg_list, regressor: str, df_columns: Sequence[str]) -> pd.DataFrame:
    values = []
    for reg_result in reg_list:
        try:
            value = reg_result.tvalues[regressor]
        except KeyError:
            value = np.nan
        values.append(value)
    df = pd.DataFrame(
        data=[tuple(values)],
        columns=df_columns,
        index=['']
    )
    return df


def convert_to_stderr_format(value: float, num_decimals: int = 2) -> str:
    if pd.isnull(value):
        return ''
    return f'({value:.{num_decimals}f})'
