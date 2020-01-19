from regtools.summarize.yesno import _standardize_col_boolean_dict, _add_yes_no_row


def add_cluster_rows(summ_df, cluster_dict):
    """
    summ_df: regression summary df where models are columns
    cluster_dict: dictionary where keys are names of cluster variables and values are a single
    boolean for whether to include clustering for all models, or a list of booleans for
    whether to include clustering for each model, e.g. for a 3 model summ_df:
        cluster_dict = {
            'Industry': True,
            'Time': [False, False, True]
        }
    """

    standard_cluster_dict = _standardize_col_boolean_dict(summ_df, cluster_dict)

    for cluster_var, booleans in standard_cluster_dict.items():
        summ_df = _add_cluster_row(summ_df, booleans, cluster_var)

    return summ_df


def _add_cluster_row(summ_df, bool_list, cluster_variable='Month'):
    return _add_yes_no_row(summ_df, bool_list, item_name=f'Cluster by {cluster_variable}')
