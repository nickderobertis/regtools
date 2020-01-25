from typing import Sequence, Union, Optional, Dict
import pandas as pd

from regtools.regtypes import LinearModelsKwargs
from regtools.tools import _to_list_if_not


def _estimate_handling_robust_and_cluster(regdf, model, robust, cluster, **fit_kwargs):
    assert not (robust and cluster)  # need to pick one of robust or cluster

    if cluster:
        cluster_fit_kwargs = _linearmodels_fit_kwarg_dict_from_cluster(cluster, regdf)
        return model.fit(**cluster_fit_kwargs, **fit_kwargs)

    if robust:
        return model.fit(cov_type="robust", **fit_kwargs)

    return model.fit(**fit_kwargs)


def _linearmodels_fit_kwarg_dict_from_cluster(
    cluster: Optional[Union[Sequence[str], str]], regdf: pd.DataFrame
) -> LinearModelsKwargs:
    cluster = _to_list_if_not(cluster)
    if len(cluster) > 2:
        raise ValueError(
            "cannot do more than two way clustering in linearmodels backend. use "
            f"statsmodels backend for 3+ way clustering. got clusters {cluster}"
        )

    fit_kwargs: Dict[str, Union[str, bool, pd.DataFrame]] = dict(cov_type="clustered")
    entity_col, time_col = regdf.index.names
    other_cols = []
    for col in cluster:
        if col == entity_col:
            fit_kwargs.update(dict(cluster_entity=True))
        elif col == time_col:
            fit_kwargs.update(dict(cluster_time=True))
        else:
            other_cols.append(col)
    if other_cols:
        fit_kwargs.update(dict(clusters=regdf[other_cols]))

    return fit_kwargs

