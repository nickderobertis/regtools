from typing import Tuple, Optional
import pandas as pd

from regtools.regtypes import (
    StrOrListOfStrs,
    InteractionTuples,
    StrOrBool,
    StrOrListOfStrsOrNone,
)
from regtools.lag.main import (
    create_lagged_variables_return_yvars_xvars_interaction_tuples,
)
from regtools.dataprep import (
    _set_interaction_tuples,
    _collect_all_variables_from_xvars_and_interaction_tuples,
    _drop_missings_df,
    _y_X_from_df,
    _set_fe,
)

YvarXvars = Tuple[str, StrOrListOfStrs]
DfYvarXvars = Tuple[pd.DataFrame, str, StrOrListOfStrs]
DfYvarXvarsLagvars = Tuple[pd.DataFrame, str, StrOrListOfStrs, StrOrListOfStrs]


def _create_reg_df_y_x_and_lag_vars(
    df: pd.DataFrame,
    yvar: str,
    xvars: StrOrListOfStrs,
    entity_var: str,
    time_var: str,
    cluster=False,
    cons=True,
    fe=None,
    interaction_tuples=None,
    num_lags=0,
    lag_variables="xvars",
    lag_period_var="Date",
    lag_id_var="TICKER",
    fill_method: Optional[str] = "ffill",
    fill_limit: int = None,
) -> DfYvarXvarsLagvars:

    # Handle lags
    (
        df,
        reg_yvar,
        reg_xvars,
        interaction_tuples,
        lag_variables,
    ) = create_lagged_variables_return_yvars_xvars_interaction_tuples(
        df,
        yvar,
        xvars,
        interaction_tuples=interaction_tuples,
        num_lags=num_lags,
        lag_variables=lag_variables,
        lag_period_var=lag_period_var,
        lag_id_var=lag_id_var,
        fill_method=fill_method,
        fill_limit=fill_limit,
    )

    fe = _set_fe(fe)
    interaction_tuples = _set_interaction_tuples(interaction_tuples)
    regdf, y, X = _get_reg_df_y_x(
        df,
        reg_yvar,
        reg_xvars,
        entity_var,
        time_var,
        cluster,
        cons,
        fe,
        interaction_tuples,
    )
    return regdf, y, X, lag_variables


def _get_reg_df_y_x(
    df: pd.DataFrame,
    yvar: str,
    xvars: StrOrListOfStrs,
    entity_var: str,
    time_var: str,
    cluster: StrOrBool,
    cons: bool,
    fe: StrOrListOfStrsOrNone,
    interaction_tuples: InteractionTuples,
) -> DfYvarXvars:
    all_xvars = _collect_all_variables_from_xvars_and_interaction_tuples(
        xvars, interaction_tuples
    )
    regdf = _drop_missings_df(df, yvar, all_xvars, cluster, fe)
    regdf = regdf.set_index([entity_var, time_var])
    # put None for fe as fe will be handled by linearmodels
    y, X, _ = _y_X_from_df(regdf, yvar, xvars, cons, None, interaction_tuples)

    return regdf, y, X
