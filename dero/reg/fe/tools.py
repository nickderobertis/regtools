
def extract_all_dummy_cols_from_dummy_cols_dict(dummy_cols_dict):
    all_values = []
    for fe_col, fe_values in dummy_cols_dict.items():
        all_values += fe_values
    return all_values