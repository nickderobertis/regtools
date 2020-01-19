from regtools.summarize.yesno import _standardize_col_boolean_dict, _add_yes_no_row


def add_fixed_effects_rows(summ_df, fixed_effect_dict):
    """
    summ_df: regression summary df where models are columns
    fixed_effect_dict: dictionary where keys are names of fixed effects and values are a single
    boolean for whether to include the fixed effect for all models, or a list of booleans for
    whether to include the fixed effect for each model, e.g. for a 3 model summ_df:
        fixed_effect_dict = {
            'Industry': True,
            'Time': [False, False, True]
        }
    """

    fe_dict = _standardize_col_boolean_dict(summ_df, fixed_effect_dict)

    for fixed_effect_name, booleans in fe_dict.items():
        summ_df = _add_fixed_effects_row(summ_df, booleans, fixed_effect_name)

    return summ_df


def _add_fixed_effects_row(summ_df, bool_list, fixed_effect_name='Industry'):
    return _add_yes_no_row(summ_df, bool_list, item_name=f'{fixed_effect_name} Fixed Effects')

