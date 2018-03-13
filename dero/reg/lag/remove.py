import pandas as pd

class SimplifiedRegressionResult:
    def __init__(self, params, pvalues, nobs, rsquared_adj, tvalues, bse, conf_int, exog_names, endog_names):
        self.params = params
        self.pvalues = pvalues
        self.nobs = nobs
        self.rsquared_adj = rsquared_adj
        self.tvalues = tvalues
        self.bse = bse
        self.conf_int = conf_int
        self.model = SimplifiedModel(exog_names, endog_names)

    @classmethod
    def from_statsmodels_result(cls, result):

        # Get direct attributes
        pull_attrs = ['params', 'pvalues', 'tvalues', 'nobs', 'rsquared_adj', 'bse', 'conf_int']
        result_dict = _extract_attrs_into_dict(result, pull_attrs)

        # Get attributes of model
        model_attrs = ['exog_names', 'endog_names']
        result_dict.update(_extract_attrs_into_dict(result.model, model_attrs))

        return cls(**result_dict)

class SimplifiedModel:

    def __init__(self, exog_names, endog_names):
        self.exog_names = exog_names
        self.endog_names = endog_names

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
    [_remove_lag_names_from_series_index(getattr(result, item), lags=lags) for item in ('params', 'pvalues', 'tvalues')]

    # Modify model properties and reassign (functions not inplace)
    result.model.endog_names = _remove_lag_names_from_varname(result.model.endog_names, lags=lags)
    result.model.exog_names = _remove_lag_names_from_list(result.model.exog_names, lags=lags)

    return result


def _remove_lag_names_from_series_index(series, lags=(1,)):
    """
    Note: inplace
    """
    [_remove_one_lag_names_from_series_index(series, num_lags=num_lags) for num_lags in lags]


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
