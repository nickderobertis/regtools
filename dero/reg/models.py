import statsmodels.api as sm
from linearmodels import FamaMacBeth

from dero.reg.linmodels.bindings.modelstr import _is_fama_macbeth_str


def get_model_class_by_string(model_string):
    if _is_logit_str(model_string):
        return sm.Logit
    elif _is_probit_str(model_string):
        return sm.Probit
    elif _is_ols_str(model_string):
        return sm.OLS
    elif _is_fama_macbeth_str(model_string):
        return FamaMacBeth
    else:
        raise ValueError(f'model string does not signify ols, probit, or logit. got {model_string}')


def _is_probit_str(model_string):
    return model_string.lower().strip() in ('probit','p','prob','prbt')

def _is_logit_str(model_string):
    return model_string.lower().strip() in ('logit','l','log','lgt')

def _is_ols_str(model_string):
    return model_string.lower().strip() in ('ols','o','reg','least squares', 'ordinary least squares')

