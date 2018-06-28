from typing import Tuple
import pandas as pd
import statsmodels.api as sm

from dero.reg.regtypes import StrOrListOfStrs, InteractionTuples, StrOrBool, StrOrListOfStrsOrNone
from dero.reg.lag.main import create_lagged_variables_return_yvars_xvars_interaction_tuples
from dero.reg.dataprep import (
    _set_interaction_tuples,
    _collect_all_variables_from_xvars_and_interaction_tuples,
    _drop_missings_df,
    create_interaction_variables
)

YvarXvars = Tuple[str, StrOrListOfStrs]
DfYvarXvars = Tuple[pd.DataFrame, str, StrOrListOfStrs]
DfYvarXvarsLagvars = Tuple[pd.DataFrame, str, StrOrListOfStrs, StrOrListOfStrs]

def _create_reg_df_y_x_and_lag_vars(df: pd.DataFrame, yvar: str, xvars: StrOrListOfStrs,
                                    entity_var: str, time_var: str,
                                    cluster=False, cons=True, fe=None, interaction_tuples=None,
                                   num_lags=0, lag_variables='xvars', lag_period_var='Date', lag_id_var='TICKER'
                                    ) -> DfYvarXvarsLagvars:

    # TODO: implement FE
    if fe is not None:
        raise NotImplementedError('need to implement FE with linearmodels PanelOLS')

    # Handle lags
    df, reg_yvar, reg_xvars, interaction_tuples, lag_variables = create_lagged_variables_return_yvars_xvars_interaction_tuples(
        df, yvar, xvars,
        interaction_tuples=interaction_tuples,
        num_lags=num_lags,
        lag_variables=lag_variables,
        lag_period_var=lag_period_var,
        lag_id_var=lag_id_var
    )

    interaction_tuples = _set_interaction_tuples(interaction_tuples)
    regdf, y, X = _get_reg_df_y_x(df, reg_yvar, reg_xvars, entity_var, time_var, cluster, cons, fe, interaction_tuples)
    return regdf, y, X, lag_variables

def _get_reg_df_y_x(df: pd.DataFrame, yvar: str, xvars: StrOrListOfStrs, entity_var: str, time_var: str,
                    cluster: StrOrBool,
                    cons: bool, fe: StrOrListOfStrsOrNone, interaction_tuples: InteractionTuples) -> DfYvarXvars:
    all_xvars = _collect_all_variables_from_xvars_and_interaction_tuples(xvars, interaction_tuples)
    regdf = _drop_missings_df(df, yvar, all_xvars, cluster, fe)
    regdf = regdf.set_index([entity_var, time_var])
    y, X = _y_X_from_df(regdf, yvar, xvars, cons, interaction_tuples)

    return regdf, y, X

def _y_X_from_df(regdf: pd.DataFrame, yvar: str, xvars: StrOrListOfStrs,
                 cons: bool, interaction_tuples: InteractionTuples) -> YvarXvars:

    model_xvars = xvars.copy()

    if interaction_tuples:
        interaction_vars = create_interaction_variables(regdf, interaction_tuples)
        model_xvars += interaction_vars

    y = regdf[yvar]
    X = regdf.loc[:, model_xvars]

    if cons:
        X = sm.add_constant(X)

    return y, X

# def _validate_reg_inputs(**reg_kwargs):
