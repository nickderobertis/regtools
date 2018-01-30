import pandas as pd


from ..tools import _to_list_if_str

def fixed_effects_reg_df_and_cols_dict(df, fe_vars):
    fe_vars = _to_list_if_str(fe_vars)

    fe_cols_dict = {}
    for fe_var in fe_vars:
        df, cols = _fixed_effects_reg_df_and_cols(df, fe_var)
        fe_cols_dict[fe_var] = cols

    return df, fe_cols_dict

def _fixed_effects_reg_df_and_cols(df, fe_var):
    dummies = pd.get_dummies(df[fe_var].astype(str)).iloc[:,1:] #drop first dummy, this will be the excluded group
    dummy_cols = [col for col in dummies.columns]
    return pd.concat([df, dummies], axis=1).drop(fe_var, axis=1), dummy_cols

