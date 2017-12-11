import pandas as pd

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

    fe_dict = _standardize_fixed_effects_dict(summ_df, fixed_effect_dict)

    for fixed_effect_name, booleans in fe_dict.items():
        summ_df = _add_fixed_effects_row(summ_df, booleans, fixed_effect_name)

    return summ_df


def _standardize_fixed_effects_dict(summ_df, fixed_effect_dict):
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
            if (not isinstance(booleans, list)):
                booleans = [booleans]
            if len(booleans) > 1:
                raise ValueError(
                    f'Incorrect shape of booleans for {fixed_effect_name} fixed effect passed. Got {len(booleans)} bools, was expecting {num_models}')
            out_booleans = [booleans[0]] * num_models

            # Final input checks
        assert isinstance(out_booleans, list)
        assert all([isinstance(b, bool) for b in out_booleans])
        assert isinstance(fixed_effect_name, str)

        # Assign to output dictionary
        out_dict[fixed_effect_name] = out_booleans

    return out_dict


def _add_fixed_effects_row(summ_df, bool_list, fixed_effect_name='Industry'):
    fixed_effects_row = pd.DataFrame([_get_yes_no(bool_) for bool_ in bool_list], columns=[fixed_effect_name + ' Fixed Effects']).T
    fixed_effects_row.columns = summ_df.columns
    return pd.concat([summ_df, fixed_effects_row], axis=0)

def _get_yes_no(bool_):
    if bool_:
        return 'Yes'
    else:
        return 'No'