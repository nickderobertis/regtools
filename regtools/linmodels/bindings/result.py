from linearmodels.panel.results import PanelResults


class ModelData:

    def __init__(self, param_names):
        self.param_names = param_names


def _convert_linearmodels_result_to_statsmodels_result_format(result: PanelResults) -> None:
    """
    Note: inplace
    """

    # First at the result level
    move_dict = {
        'std_errors': 'bse',
        'tstats': 'tvalues'
    }
    for old_attr, new_attr in move_dict.items():
        _move_attr(result, old_attr, new_attr)

    # Now at the model level
    # Create result.model.data.param_names
    result.model.data = ModelData(
        param_names=[col for col in result.params.index],
    )
    result.model.endog_names = result.model.dependent.vars[0]


def _move_attr(obj: any, old_attr: str, new_attr: str) -> None:
    """
    Note: inplace
    """
    attr_value = getattr(obj, old_attr)
    setattr(obj, new_attr, attr_value)
    try:
        delattr(obj, old_attr)
    except AttributeError:
        pass