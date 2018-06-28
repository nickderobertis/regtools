


def _is_fama_macbeth_str(model_string: str) -> bool:
    simple_str = model_string.lower().strip().replace('-','').replace('_','').replace(' ','')
    return simple_str in ('famamacbeth', 'fmb', 'fb', 'fm', 'fmacbeth','famam','fmbreg','fmreg')

# Note: add model str functions here as they are created. This will feed into _is_linearmodels_str
model_str_funcs = [
    _is_fama_macbeth_str
]

def _is_linearmodels_str(model_string: str) -> bool:
    return any([is_model_str_func(model_string) for is_model_str_func in model_str_funcs])