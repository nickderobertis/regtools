import pandas as pd
import matplotlib.pyplot as plt

from dero.reg.interact import _interaction_tuple_to_var_name
from dero.reg.iter import _set_interaction_tuples
from dero.reg.hypothesis.lincom import hypothesis_test

# TODO: make module flexible to not having interaction

def interacted_lag_plot_from_reg_result_list(results, lag_tuple, main_iv, interaction_tuples, yvar,
                                             interaction_var_value=13.27, date_var='Year',
                                             outpath=None, clear_figure=True):
    plot_df = produce_simplified_result_df(
        results, lag_tuple, main_iv, interaction_tuples, interaction_var_value=interaction_var_value
    )
    return interacted_lag_plot_from_lag_result_df(
        plot_df, yvar, main_iv, date_var=date_var, outpath=outpath, clear_figure=clear_figure
    )

def interacted_lag_plot_from_lag_result_df(result_df, yvar, main_iv, date_var='Year', outpath=None, clear_figure=True):
    """
    Creates a plot of effect of main_iv on yvar at different quantiles. To be used after
    reg_for_each_quantile_produce_result_df

    :param result_df: pd.DataFrame, result from reg_for_each_quantile_produce_result_df
    :param yvar: str, label of dependent variable
    :param main_iv: str, label of independent variable of interest
    :param outpath: str, filepath to output figure. must include matplotlib supported extension such as .pdf or .png
    :param clear_figure: bool, True wipe memory of matplotlib figure after running function
    :return:
    """
    p1 = plt.plot(result_df.q, result_df.b, color='black', label='Lag Regression Slope')
    p2 = plt.fill_between(result_df.q, result_df.ub, result_df.lb, color='gray',
                          label='95% Confidence Interval Bound')
    plt.title(f'Effect of {main_iv} on {yvar} over Time')
    plt.ylabel(r'Lag Coefficient')
    plt.xlabel(f'Number of {date_var}s Lagged')
    plt.legend()

    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()

    if clear_figure:
        plt.clf()

def produce_simplified_result_df(results, lag_tuple, main_iv, interaction_tuples, interaction_var_value=13.27):
    simple_results = _produce_simplified_results(
        results, lag_tuple, main_iv, interaction_tuples, interaction_var_value=interaction_var_value
    )
    return pd.DataFrame(simple_results, columns=['q', 'a', 'b', 'lb', 'ub'])


def _produce_simplified_results(results, lag_tuple, main_iv, interaction_tuples, interaction_var_value=13.27):
    interaction_tuples = _set_interaction_tuples(interaction_tuples, len(results))
    return [
        _produce_simplified_result_list_from_one_result(
            result, lag_tuple[i], main_iv, interaction_tuples[i], interaction_var_value=interaction_var_value
        ) for i, result in enumerate(results)
    ]


def _produce_simplified_result_list_from_one_result(res, lag, main_iv, interaction_tuple, interaction_var_value=13.27):
    # Handle possibility of dummy cols dict coming through with result as tuple
    if isinstance(res, tuple):
        res = res[0]

    interaction_name = _interaction_tuple_to_var_name(interaction_tuple)

    # Just reassign names for brevity
    b_iv = res.params[main_iv]
    b_interact = res.params[interaction_name]

    # 95% confidence interval +/- amount
    # Get standard error of linear combination
    col_coef_dict = {
        main_iv: 1,
        interaction_name: interaction_var_value
    }
    hypothesis_result = hypothesis_test(res, col_coef_dict=col_coef_dict)

    return [
        lag,
        res.params['const'],
        hypothesis_result.effect,
        hypothesis_result.conf_int()[0][0],  # lower 95% confidence interval
        hypothesis_result.conf_int()[0][1],  # upper 95% confidence interval
    ]