from statsmodels.iolib.summary2 import summary_col
from .fe import add_fixed_effects_rows
from .fe.tools import extract_all_dummy_cols_from_dummy_cols_dict_list, \
    extract_all_fe_names_from_dummy_cols_dict_list



def produce_summary(reg_sets, stderr=False, float_format='%0.1f'):
    reg_list, dummy_col_dicts = _extract_result_list_and_dummy_dicts(reg_sets)

    summ = summary_col(reg_list, stars=True, float_format=float_format,
                       info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                  'R2': lambda x: "{:.2f}".format(x.rsquared),
                                  'Adj-R2': lambda x: "{:.2f}".format(x.rsquared_adj)})

    # TODO: handle removing stderr for FE when stderr=True

    # Handle fe - remove individual fe cols and replace with e.g. Industry Fixed Effects No, Yes, Yes
    all_cols_to_remove = extract_all_dummy_cols_from_dummy_cols_dict_list(dummy_col_dicts)
    summ.tables[0].drop(all_cols_to_remove, axis=0, inplace=True)
    fe_dict = _multiple_model_fe_dict_from_dummy_col_dict_list(dummy_col_dicts)
    summ.tables[0] = add_fixed_effects_rows(summ.tables[0], fe_dict)

    if not stderr:
        summ.tables[0].drop('', axis=0, inplace=True)  # drops the rows containing standard errors

    return summ


def _extract_result_list_and_dummy_dicts(result_sets):
    plain_results = []
    dummy_dicts = []
    for ambiguous_result in result_sets:
        # This is the case where fe has been passed, and we have a dummy_col_dict
        if isinstance(ambiguous_result, tuple):
            plain_results.append(ambiguous_result[0])
            dummy_dicts.append(ambiguous_result[1])
        # No fe passed, just plain result
        else:
            plain_results.append(ambiguous_result)
            dummy_dicts.append(None)  # keep order by appending None

    return plain_results, dummy_dicts


def _multiple_model_fe_dict_from_dummy_col_dict_list(dummy_col_dict_list):
    fixed_effect_rows = extract_all_fe_names_from_dummy_cols_dict_list(dummy_col_dict_list)
    out_dict = {fe_name: [] for fe_name in fixed_effect_rows}
    for dummy_col_dict in dummy_col_dict_list:
        for fe_name in fixed_effect_rows:
            if (not dummy_col_dict) or (fe_name not in dummy_col_dict):
                out_dict[fe_name].append(False)
            else:
                out_dict[fe_name].append(True)

    return out_dict