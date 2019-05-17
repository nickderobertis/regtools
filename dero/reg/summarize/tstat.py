from typing import Tuple, List, Sequence
import pandas as pd
import numpy as np
from dero.data.formatters.stars import parse_stars_value
from dero.data.formatters.stderr import parse_stderr_value, convert_to_stderr_format
from dero.reg.summarize.split import get_var_df_and_non_var_df


def replace_stderr_with_t_stat_in_summary_df(df: pd.DataFrame, split_rows: Sequence[str]) -> pd.DataFrame:
    var_df, non_var_df = get_var_df_and_non_var_df(df, split_rows=split_rows)

    var_df = replace_stderr_with_t_stat_in_var_df(var_df)

    return pd.concat([var_df, non_var_df], axis=0)


def replace_stderr_with_t_stat_in_var_df(df: pd.DataFrame) -> pd.DataFrame:
    # Create column identifying row as an estimate or standard error
    df['type'] = ['estimate', 'stderr'] * int(len(df.index) / 2)

    # Create column identifying variable name of row (no spaces)
    df['regressor'] = [i for sublist in [[j] * 2 for j in df.index[0::2]] for i in sublist]

    for regressor in df['regressor'].unique():
        numeric_cols = [col for col in df.columns if col not in ['regressor', 'type']]
        coefs, stderrs = _get_coef_and_stderr_series_from_modified_summary_df(df, regressor, numeric_cols)
        t_values = coefs / stderrs
        t_values.index = ['']
        t_values = t_values.applymap(convert_to_stderr_format)
        df.loc[
            (df['regressor'] == regressor) &
            (df['type'] == 'stderr'),
            numeric_cols
        ] = t_values

    # Delete the created columns
    df.drop(['type', 'regressor'], axis=1, inplace=True)

    return df


def _get_coef_and_stderr_series_from_modified_summary_df(df: pd.DataFrame, regressor: str,
                                                         numeric_cols: List[str]) -> Tuple[pd.Series, pd.Series]:

    stderrs = df.loc[
        (df['regressor'] == regressor) &
        (df['type'] == 'stderr'),
        numeric_cols
    ].applymap(parse_stderr_value).reset_index(drop=True)

    coefs = df.loc[
        (df['regressor'] == regressor) &
        (df['type'] == 'estimate'),
        numeric_cols
    ].applymap(_parse_stars_get_coef).reset_index(drop=True)

    return coefs, stderrs


def _parse_stars_get_coef(value: str) -> float:
    result, stars = parse_stars_value(value)
    if not result:
        return np.nan
    return float(result)
