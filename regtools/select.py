from typing import Sequence, List, Any


def select_models(reg_list, keepnum: int, xvars: Sequence[str]):
    """
    Takes a list of fitted regression models and selects among them based on adjusted R-Squared. For each
    number of variables involved in the regressions, keepnum with the highest R-squareds will be kept.

    For example, if reg_list contains 3 regressions with two variables and 6 regressions with three variables,
    and keepnum is 2, will return a list of four regressions, 2 with two variables and 2 with three variables.

    :param reg_list:
    :param keepnum: number to keep for each amount of x variables. The total number of outputted
        regressions will be roughly keepnum * len(xvars)
    :param xvars: column names of x variables
    :return:
    """
    outlist: List[Any] = []
    for i in range(1, len(xvars) + 1):
        reg_list_match = [reg for reg in reg_list if reg.df_model == i] #select models with this many variables
        try:
            r2_min = sorted([reg.rsquared_adj for reg in reg_list_match])[-keepnum] #gets keepnumth highest r2
        except IndexError: #should happen once there are less models run than keepnum (i.e. with all xvars)
            r2_min = sorted([reg.rsquared_adj for reg in reg_list_match])[0] #gets lowest r2 (keep all)
        outlist += [reg for reg in reg_list_match if reg.rsquared_adj >= r2_min]
    return outlist