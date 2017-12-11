
def extract_all_dummy_cols_from_dummy_cols_dict(dummy_cols_dict):
    all_values = []
    for fe_col, fe_values in dummy_cols_dict.items():
        all_values += fe_values
    return all_values

def extract_all_dummy_cols_from_dummy_cols_dict_list(dummy_cols_dict_list):
    all_cols = []
    for dummy_cols_dict in dummy_cols_dict_list:
        if dummy_cols_dict:
            all_cols += extract_all_dummy_cols_from_dummy_cols_dict(dummy_cols_dict)
    return list(set(all_cols))


def extract_all_fe_names_from_dummy_cols_dict_list(dummy_cols_dict_list):
    names = []
    for dummy_cols_dict in dummy_cols_dict_list:
        if dummy_cols_dict:
            names += [name for name in dummy_cols_dict]

    return list(set(names))