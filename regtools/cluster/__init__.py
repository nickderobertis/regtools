from typing import List, Optional
import itertools

import pandas as pd

from regtools.ext_statsmodels import update_statsmodel_result_with_new_cov_matrix

StrOrNone = Optional[str]
StrOrNoneList = List[StrOrNone]
StrOrNoneListList = List[StrOrNoneList]


def estimate_model_handling_cluster(
    regdf: pd.DataFrame, model, cluster: List[str], **fit_kwargs
):
    """
    Handles multiway clustering through multiple estimations following
    Cameron, Gelbach, and Miller (2011).
    """
    cluster_groups = _multiway_cluster_groups(cluster)

    if len(cluster_groups) == 0:
        raise ValueError(
            f"did not get any cluster groups, yet cluster was called with {cluster}"
        )

    for i, cluster_group in enumerate(cluster_groups):
        result = _estimate_model_handling_single_cluster(
            regdf, model, cluster_group, **fit_kwargs
        )
        cluster_group_cov_matrix = result.cov_params()
        if i == 0:
            cov_matrix = cluster_group_cov_matrix
        else:
            # Handle combining the covariance matrices across the different cluster estimations
            # Follow eq 2.13 in CGM (2011), where odd number of cluster groups are added
            # and even number of cluster groups are subtracted
            sign = _negative_one_if_even_positive_one_if_odd(len(cluster_group))
            cov_matrix = cov_matrix + (sign * cluster_group_cov_matrix)

    # All parameter estimates should be identical, so can just override last result's cov matrix to
    # get final result
    update_statsmodel_result_with_new_cov_matrix(result, cov_matrix)

    return result


def _estimate_model_handling_single_cluster(
    regdf: pd.DataFrame, model, cluster: List[str], **fit_kwargs
):
    cluster_ids = _cluster_group_id_series(regdf, cluster)
    result = model.fit(
        cov_type="cluster", cov_kwds={"groups": cluster_ids}, **fit_kwargs
    )
    return result


def _multiway_cluster_groups(cluster_vars: List[str]) -> List[List[str]]:
    """
    Transforms cluster_vars into the sets of cluster variables on which to run individual
    regressions, following Cameron, Gelbach, and Miller (2011).
    """

    cluster_vectors = _cluster_vars_to_cluster_vector_lists(cluster_vars)

    all_cluster_groups = []
    for group_tuple in itertools.product(*cluster_vectors):
        # group_tuple may come with Nones, such as ('Firm', None), or (None, None)
        # we only want to extract the non Nones
        valid_items = tuple([item for item in group_tuple if item is not None])
        if len(valid_items) > 0:
            all_cluster_groups.append(valid_items)

    # Remove duplicates and convert tuples to lists
    all_cluster_groups_lists = [list(group) for group in set(all_cluster_groups)]

    return all_cluster_groups_lists


def _cluster_vars_to_cluster_vector_lists(cluster_vars: List[str]) -> StrOrNoneListList:
    """
    Transforms cluster_vars into a format which can be used with itertools.product.

    E.g. cluster_vars = ['Firm', 'Date'] -> [
        ['Firm', None],
        [None, 'Date']
    ]

    and cluster_vars = ['Firm', 'Date', 'Portfolio'] -> [
        ['Firm', None, None],
        [None, 'Date', None],
        [None, None, 'Portfolio']
    ]
    """

    num_items = len(cluster_vars)
    all_lists: StrOrNoneListList = []
    for i, cluster_var in enumerate(cluster_vars):
        output_list = [None] * num_items
        output_list[i] = cluster_var  # type: ignore
        all_lists.append(output_list)  # type: ignore

    return all_lists


def _cluster_group_id_series(df: pd.DataFrame, cluster_vars: List[str]) -> pd.Series:
    unique_groups = df[cluster_vars].drop_duplicates()
    unique_groups["_group_id"] = range(0, len(unique_groups))
    return df[cluster_vars].merge(unique_groups, how="left", on=cluster_vars)[
        "_group_id"
    ]


def _negative_one_if_even_positive_one_if_odd(num: int) -> int:
    if _is_even(num):
        return -1
    else:
        return 1


def _is_even(num: int) -> bool:
    return num % 2 == 0
