import statsmodels.api as sm

from regtools.linmodels.bindings.modelstr import _is_fama_macbeth_str


def get_model_class_by_string(model_string):
    if _is_logit_str(model_string):
        return sm.Logit
    elif _is_probit_str(model_string):
        return sm.Probit
    elif _is_ols_str(model_string):
        return sm.OLS
    else:
        raise ValueError(f'model string does not signify ols, probit, logit, or fama macbeth. got {model_string}')

def get_model_name_by_string(model_string: str) -> str:
    if _is_logit_str(model_string):
        return 'Logit'
    elif _is_probit_str(model_string):
        return 'Probit'
    elif _is_ols_str(model_string):
        return 'OLS'
    elif _is_fama_macbeth_str(model_string):
        return 'Fama-Macbeth'
    else:
        raise ValueError(f'model string does not signify ols, probit, logit, or fama macbeth. got {model_string}')


def _is_probit_str(model_string):
    return model_string.lower().strip() in ('probit','p','prob','prbt')

def _is_logit_str(model_string):
    return model_string.lower().strip() in ('logit','l','log','lgt')

def _is_ols_str(model_string):
    return model_string.lower().strip() in ('ols','o','reg','least squares', 'ordinary least squares')

# Note: add model str functions here as they are created. This will feed into _is_linearmodels_str
model_str_funcs = [
    _is_ols_str,
    _is_logit_str,
    _is_probit_str
]

def _is_regular_reg_str(model_string: str) -> bool:
    return any([is_model_str_func(model_string) for is_model_str_func in model_str_funcs])