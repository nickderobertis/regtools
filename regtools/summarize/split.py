from typing import Sequence
import pandas as pd


def get_var_df_and_non_var_df(df: pd.DataFrame, split_rows: Sequence[str] = ('N', 'R2', 'Adj-R2')):
    """
    Splits rows containing N, R2, Adj-R2 into separate dataframe
    """
    other_data_mask = df.index.isin(split_rows)
    return df.loc[~other_data_mask], df.loc[other_data_mask]
