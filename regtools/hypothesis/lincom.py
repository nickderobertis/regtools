import numpy as np

def hypothesis_test(result, col_coef_dict):
    """
    result: statsmodels RegressionResult or dero SimplifiedRegressionResult
    col_coef_dict: dict, keys are names of columns in DataFrame used for regression. Values are
                   coefficients to apply to those columns.
    """
    r_matrix = _create_r_matrix_for_hypothesis_test(result, col_coef_dict)
    return result.t_test(r_matrix, use_t=True)


def _create_r_matrix_for_hypothesis_test(result, col_coef_dict):
    """
    col_coef_dict: dict, keys are names of columns in DataFrame used for regression. Values are
                   coefficients to apply to those columns.
    """
    index_coef_dict = _replace_dict_keys_with_key_index_in_list(col_coef_dict, result.model.exog_names)
    r_matrix = _numpy_array_from_index_coef_dict(result, index_coef_dict)

    return r_matrix


def _numpy_array_from_index_coef_dict(result, index_coef_dict):
    arr = np.zeros(len(result.model.exog_names))
    _modify_array_from_index_coef_dict(arr, index_coef_dict)

    return arr


def _modify_array_from_index_coef_dict(arr, index_coef_dict):
    """
    Note: inplace
    """
    for index, coef in index_coef_dict.items():
        arr[index] = coef


def _replace_dict_keys_with_key_index_in_list(dict_, list_):
    return {list_.index(key): value for key, value in dict_.items()}