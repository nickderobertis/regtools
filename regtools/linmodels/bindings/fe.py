from typing import Optional, Any, Sequence, Union
import pandas as pd
from linearmodels import PanelOLS

from regtools.regtypes import LinearModelsKwargs, DummyColsDict, StrOrListOfStrsOrNone
from regtools.tools import _to_list_if_not


def dummy_cols_dict_from_model(
    model: PanelOLS, regdf: pd.DataFrame
) -> Optional[DummyColsDict]:
    dummy_cols_dict = {}
    firm_index, time_index = regdf.index.levels
    if _has_attr_and_attr_is_truthy(model, "entity_effects"):
        dummy_cols_dict[firm_index.name] = firm_index.unique().tolist()
    if _has_attr_and_attr_is_truthy(model, "time_effects"):
        dummy_cols_dict[time_index.name] = time_index.unique().tolist()
    if _has_attr_and_attr_is_truthy(model, "_other_effect_cats") and model._other_effect_cats is not None:
        other_df = model._other_effect_cats.dataframe
        other_dict = {col: other_df[col].unique().tolist() for col in other_df.columns}
        dummy_cols_dict.update(other_dict)

    return dummy_cols_dict


def linearmodels_fe_kwarg_dict_from_fe(
    fe: Optional[Union[str, Sequence[str]]], regdf: pd.DataFrame
) -> LinearModelsKwargs:
    if fe is None:
        return {}

    fe = _to_list_if_not(fe)

    fe_kwargs: LinearModelsKwargs = {}
    entity_col, time_col = regdf.index.names
    other_cols = []
    for col in fe:
        if col == entity_col:
            fe_kwargs.update(dict(entity_effects=True))
        elif col == time_col:
            fe_kwargs.update(dict(time_effects=True))
        else:
            other_cols.append(col)
    if other_cols:
        fe_kwargs.update(dict(other_effects=regdf[other_cols]))

    return fe_kwargs


def _has_attr_and_attr_is_truthy(obj: Any, attr: str) -> bool:
    if not hasattr(obj, attr):
        return False
    value = getattr(obj, attr)
    return value is not None and value is not False and value != 0
