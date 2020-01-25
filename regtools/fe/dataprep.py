import pandas as pd
from typing import Tuple, Optional, List, Union

from ..tools import _to_list_if_str

DfListTuple = Tuple[pd.DataFrame, Optional[list]]


def fixed_effects_reg_df_and_cols_dict(df, fe_vars):
    fe_vars = _to_list_if_str(fe_vars)

    fe_cols_dict = {}
    for fe_var in fe_vars:
        df, cols = _fixed_effects_reg_df_and_cols(df, fe_var)
        fe_cols_dict[fe_var] = cols

    return df, fe_cols_dict


def _fixed_effects_reg_df_and_cols(df, fe_var):
    dummies = _get_dummy_df(df, fe_var)
    dummy_cols = [col for col in dummies.columns]
    fe_df = pd.concat([df, dummies], axis=1)
    if fe_var in fe_df.columns:
        fe_df.drop(fe_var, axis=1, inplace=True)
    return fe_df, dummy_cols


def _get_dummy_df(df: pd.DataFrame, fe_var: str) -> pd.DataFrame:
    dummy_calc_df, index_cols = _get_dummy_calc_df(df, fe_var)
    dummies = pd.get_dummies(dummy_calc_df[fe_var].astype(str))
    if index_cols is not None:
        # meed to add index back to dummy df
        dummies = pd.concat([dummy_calc_df[index_cols], dummies], axis=1)
        dummies.set_index(index_cols, inplace=True)

    dummies = dummies.iloc[:, 1:]  # drop first dummy, this will be the excluded group
    return dummies


def _get_dummy_calc_df(df: pd.DataFrame, fe_var: str) -> DfListTuple:
    index_cols: Optional[List[Union[str, float, int]]]
    if fe_var in df.index.names:
        # Won't work with fe_var in index. Need to reset index for calculation
        index_cols = [col for col in df.index.names]
        for_calc_df = df.reset_index()
    elif fe_var in df.columns:
        for_calc_df = df  # fe_var is in columns, no extra processing needed before calculating dummies
        index_cols = None
    else:
        raise ValueError(
            f"fixed effects variable {fe_var} must be in columns or index."
        )

    return for_calc_df, index_cols
