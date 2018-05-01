import pandas as pd
from functools import partial
from statsmodels.api import OLS

class SimplifiedBase:
    def _set_attrs(self, attr_dict):
        for attr in attr_dict:
            setattr(self, attr, attr_dict[attr])

class SimplifiedRegressionResult(SimplifiedBase):
    direct_attrs = [
            'params', 'pvalues', 'tvalues', 'nobs', 'rsquared_adj', 'bse', 'conf_int',
            'normalized_cov_params', 'cov_params_default', 'scale', 'cov_params',
            't_test'
    ]
    model_attrs = ['exog_names', 'endog_names']


    def __init__(self, **kwargs):
        _validate_attrs(kwargs, self.direct_attrs + self.model_attrs)
        model_kwargs = SimplifiedRegressionResult.pop_model_attrs(kwargs)
        self._set_attrs(kwargs)
        self.model = SimplifiedModel(**model_kwargs)

    @classmethod
    def from_statsmodels_result(cls, result):

        # Get direct attributes
        result_dict = _extract_attrs_into_dict(result, cls.direct_attrs)

        # Get attributes of model
        result_dict.update(_extract_attrs_into_dict(result.model, cls.model_attrs))

        return cls(**result_dict)

    @classmethod
    def pop_model_attrs(cls, attr_dict):
        """
        Note: pops from attr_dict inplace
        """
        outdict = {}
        for attr in attr_dict:
            if attr in cls.model_attrs:
                outdict[attr] = attr_dict[attr]

        # Must pop separately as cannot change size of iterating dict
        [attr_dict.pop(attr) for attr in outdict]
        return outdict


class SimplifiedModel(SimplifiedBase):

    def __init__(self, **kwargs):
        self._set_attrs(kwargs)

class UnsupportedResultAttributeException(Exception):
    pass

def _validate_attrs(attr_dict, valid_attrs):
    for attr in attr_dict:
        if attr not in valid_attrs:
            raise UnsupportedResultAttributeException(f'Attribute {attr} not supported for SimplifiedRegressionResult')

def _extract_attrs_into_dict(obj, attrs):
    result_dict = {}

    # Get direct attributes
    for attr in attrs:
        value = getattr(obj, attr)
        if isinstance(value, (pd.Series, pd.DataFrame, list, dict)):
            value = value.copy()
        result_dict[attr] = value

    return result_dict


def remove_lag_names_from_reg_results(reg_list, lags=(1,)):
    """
    Note: partially inplace
    """
    out_reg_list = []
    for ambiguous_result in reg_list:

        # Determine type of result
        if isinstance(ambiguous_result, tuple):  # Tuple of result, fe_dict
            result = ambiguous_result[0]
        else:  # just a single result
            result = ambiguous_result

        # Actually replace names
        result = _remove_lag_name_from_reg_result(result, lags=lags)

        # Add to output, depending on type of result
        if isinstance(ambiguous_result, tuple):  # Tuple of result, fe_dict
            out_reg_list.append((result, ambiguous_result[1]))
        else:  # just a single result
            out_reg_list.append(result)

    return out_reg_list


def _remove_lag_name_from_reg_result(result, lags=(1,)):
    """
    Note: partially inplace
    """
    result = SimplifiedRegressionResult.from_statsmodels_result(result)

    # Modify base properties inplace
    [
        _remove_lag_names_from_ambiguous_property(getattr(result, item), lags=lags) for item in (
            'params', 'pvalues', 'tvalues', 'bse', 'normalized_cov_params'
        )
    ]

    # Modify model properties and reassign (functions not inplace)
    for attr in ['endog_names', 'exog_names']:
        setattr(
            result.model,
            attr,
            _remove_lag_names_from_ambiguous_property(
                getattr(result.model, attr),
                lags=lags)
        )

    return result

def _remove_lag_names_from_ambiguous_property(ambiguous, lags=(1,)):
    """
    Note: Series and DataFrame operations inplace, str and list operations not inplace
    """
    if isinstance(ambiguous, pd.DataFrame):
        lag_func = partial(_remove_lag_names_from_df_index_and_columns, ambiguous)
    elif isinstance(ambiguous, pd.Series):
        lag_func = partial(_remove_lag_names_from_series_index, ambiguous)
    elif isinstance(ambiguous, str):
        lag_func = partial(_remove_lag_names_from_varname, ambiguous)
    elif isinstance(ambiguous, list):
        lag_func = partial(_remove_lag_names_from_list, ambiguous)
    else:
        raise ValueError(f'Must pass DataFrame, Series, str, or list. Got type {type(ambiguous)}')

    return lag_func(lags=lags)

def _remove_lag_names_from_df_index_and_columns(df, lags=(1,)):
    """
    Note: inplace
    """
    [_remove_one_lag_names_from_df_index_and_columns(df, num_lags=num_lags) for num_lags in lags]


def _remove_lag_names_from_series_index(series, lags=(1,)):
    """
    Note: inplace
    """
    [_remove_one_lag_names_from_series_index(series, num_lags=num_lags) for num_lags in lags]


def _remove_one_lag_names_from_df_index_and_columns(df, num_lags=1):
    """
    Note: inplace
    """
    rename_dict = {col: lag_varname_to_varname(col, num_lags=num_lags) for col in df.index}
    df.index = df.index.to_series().replace(rename_dict)
    df.columns = df.columns.to_series().replace(rename_dict)

def _remove_one_lag_names_from_series_index(series, num_lags=1):
    """
    Note: inplace
    """
    rename_dict = {col: lag_varname_to_varname(col, num_lags=num_lags) for col in series.index}
    series.index = series.index.to_series().replace(rename_dict)

def _remove_lag_names_from_list(list_, lags=(1,)):
    for lag in lags:
        list_ = _remove_one_lag_names_from_list(list_, lag)
    return list_

def _remove_one_lag_names_from_list(list_, num_lags=1):
    return [lag_varname_to_varname(item, num_lags=num_lags) for item in list_]

def _remove_lag_names_from_varname(varname, lags=(1,)):
    for lag in lags:
        varname = lag_varname_to_varname(varname, lag)
    return varname

def lag_varname_to_varname(varname, num_lags=1):
    return varname.replace(rf'_{{t - {num_lags}}}', '')
