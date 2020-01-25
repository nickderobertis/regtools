import pandas as pd
from typing import List, Dict

ColBoolDict = Dict[str, List[bool]]


def col_boolean_dict_from_list_of_lists_of_columns(
    column_list_list: List[List[str]],
) -> ColBoolDict:
    """

    transforms [
        ['a', 'b'],
        ['a'],
        ['b'],
        None,
        False,
        [],
        ['a','b']
    ]

    into {
        'a': [True, True, False, False, False, False, True],
        'b': [True, False, True, False, False, False, True]
    }
    Returns:

    """
    all_columns = []
    for column_list in column_list_list:
        if column_list:
            all_columns.extend(column_list)
    all_columns = list(set(all_columns))

    col_bool_dict: Dict[str, List[bool]] = {col: [] for col in all_columns}
    for column_list in column_list_list:
        if not column_list:
            # None, False, [] for column list. Just add False to all columns
            for column in col_bool_dict:
                col_bool_dict[column].append(False)
        else:
            # Valid column list
            for column in col_bool_dict:
                if column in column_list:
                    col_bool_dict[column].append(True)
                else:
                    col_bool_dict[column].append(False)

    return col_bool_dict


def _standardize_col_boolean_dict(summ_df, fixed_effect_dict):
    num_models = len(summ_df.columns)

    out_dict = {}
    for fixed_effect_name, booleans in fixed_effect_dict.items():
        # Here we are being passed a list of booleans matching the size of models.
        # This is the correct format so just output
        if (isinstance(booleans, list)) and (len(booleans) == num_models):
            out_booleans = booleans

            # Here we are being passed a single boolean or a list with single boolean
        # Need to expand to cover all models
        else:
            if not isinstance(booleans, list):
                booleans = [booleans]
            if len(booleans) > 1:
                raise ValueError(
                    f"Incorrect shape of booleans for {fixed_effect_name} fixed effect passed. Got {len(booleans)} bools, was expecting {num_models}"
                )
            out_booleans = [booleans[0]] * num_models

            # Final input checks
        assert isinstance(out_booleans, list)
        assert all([isinstance(b, bool) for b in out_booleans])
        assert isinstance(fixed_effect_name, str)

        # Assign to output dictionary
        out_dict[fixed_effect_name] = out_booleans

    return out_dict


def _add_yes_no_row(summ_df, bool_list, item_name="Time Fixed Effects"):
    yes_no_row = pd.DataFrame(
        [_get_yes_no(bool_) for bool_ in bool_list], columns=[item_name]
    ).T
    yes_no_row.columns = summ_df.columns
    return pd.concat([summ_df, yes_no_row], axis=0)


def _get_yes_no(bool_):
    if bool_:
        return "Yes"
    else:
        return "No"

