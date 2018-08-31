from typing import Tuple, Union
import pandas as pd
import statsmodels.api as sm

from dero.reg.regtypes import StrOrListOfStrs, InteractionTuples, StrOrBool, StrOrListOfStrsOrNone
from dero.reg.lag.main import create_lagged_variables_return_yvars_xvars_interaction_tuples
from dero.reg.dataprep import (
    _set_interaction_tuples,
    _collect_all_variables_from_xvars_and_interaction_tuples,
    _drop_missings_df,
    _y_X_from_df,
    _set_fe
)

YvarXvars = Tuple[str, StrOrListOfStrs]
DummyDictOrNone = Union[dict, None]
DfYvarXvarsDummyDict = Tuple[pd.DataFrame, str, StrOrListOfStrs, DummyDictOrNone]
DfYvarXvarsLagvarsDummyDict = Tuple[pd.DataFrame, str, StrOrListOfStrs, StrOrListOfStrs, DummyDictOrNone]

def _create_reg_df_y_x_and_lag_vars(df: pd.DataFrame, yvar: str, xvars: StrOrListOfStrs,
                                    entity_var: str, time_var: str,
                                    cluster=False, cons=True, fe=None, interaction_tuples=None,
                                   num_lags=0, lag_variables='xvars', lag_period_var='Date', lag_id_var='TICKER',
                                    fill_method: str = 'ffill'
                                    ) -> DfYvarXvarsLagvarsDummyDict:

    # Handle lags
    df, reg_yvar, reg_xvars, interaction_tuples, lag_variables = create_lagged_variables_return_yvars_xvars_interaction_tuples(
        df, yvar, xvars,
        interaction_tuples=interaction_tuples,
        num_lags=num_lags,
        lag_variables=lag_variables,
        lag_period_var=lag_period_var,
        lag_id_var=lag_id_var,
        fill_method=fill_method
    )

    fe = _set_fe(fe)
    interaction_tuples = _set_interaction_tuples(interaction_tuples)
    regdf, y, X, dummy_cols_dict = _get_reg_df_y_x(df, reg_yvar, reg_xvars, entity_var, time_var, cluster, cons, fe, interaction_tuples)
    return regdf, y, X, lag_variables, dummy_cols_dict

def _get_reg_df_y_x(df: pd.DataFrame, yvar: str, xvars: StrOrListOfStrs, entity_var: str, time_var: str,
                    cluster: StrOrBool,
                    cons: bool, fe: StrOrListOfStrsOrNone, interaction_tuples: InteractionTuples) -> DfYvarXvarsDummyDict:
    all_xvars = _collect_all_variables_from_xvars_and_interaction_tuples(xvars, interaction_tuples)
    regdf = _drop_missings_df(df, yvar, all_xvars, cluster, fe)
    regdf = regdf.set_index([entity_var, time_var])
    y, X, dummy_cols_dict = _y_X_from_df(regdf, yvar, xvars, cons, fe, interaction_tuples)

    return regdf, y, X, dummy_cols_dict