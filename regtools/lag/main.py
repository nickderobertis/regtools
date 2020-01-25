import pandas as pd
from typing import Tuple, Optional

from regtools.regtypes import InteractionTuples, StrOrListOfStrs
from .create import (
    create_lagged_variables,
    _convert_interaction_tuples,
    _convert_variable_names,
    _set_lag_variables,
)

DfYvarXvarInteractionTuplesLagvarTuple = Tuple[
    pd.DataFrame, str, StrOrListOfStrs, Optional[InteractionTuples], StrOrListOfStrs
]


def create_lagged_variables_return_yvars_xvars_interaction_tuples(
    df: pd.DataFrame,
    yvar: str,
    xvars: StrOrListOfStrs,
    interaction_tuples: InteractionTuples = None,
    num_lags: int = 0,
    lag_variables: StrOrListOfStrs = "xvars",
    lag_period_var: str = "Date",
    lag_id_var: str = "TICKER",
    fill_method: str = "ffill",
    fill_limit: int = None,
) -> DfYvarXvarInteractionTuplesLagvarTuple:
    if num_lags == 0:
        return df, yvar, xvars, interaction_tuples, lag_variables

    lag_variables = _set_lag_variables(lag_variables, yvar, xvars)
    df = create_lagged_variables(
        df,
        lag_variables,
        id_col=lag_id_var,
        date_col=lag_period_var,
        num_lags=num_lags,
        fill_method=fill_method,
        fill_limit=fill_limit,
    )
    reg_yvar, reg_xvars = _convert_variable_names(
        yvar, xvars, lag_variables, num_lags=num_lags
    )
    if interaction_tuples is not None:
        interaction_tuples = _convert_interaction_tuples(
            interaction_tuples, lag_variables, num_lags=num_lags
        )

    return df, reg_yvar, reg_xvars, interaction_tuples, lag_variables
