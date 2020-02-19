import regtools

import pandas as pd
from pandas import Timestamp
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy import nan, inf



class DataFrameTest:

    df = pd.DataFrame(
        data=[
            (1, 2, 3),
            (4, 5, nan),
            (10, 11, 100)
        ],
        columns=['y', 'x1', 'x2']
    )

    df_groups = pd.DataFrame(
        data=[
            (1, 2, 3, 'a'),
            (4, 5, nan, 'a'),
            (10, 11, 100, 'a'),
            (2, 4, 6, 'b'),
            (5, 10, nan, 'b'),
            (11, 15, 150, 'b'),
        ],
        columns=['y', 'x1', 'x2', 'group']
    )

    df_groups_no_nan = pd.DataFrame(
        data=[
            (1, 2, 3, 'a'),
            (4, 5, 8, 'a'),
            (10, 11, 100, 'a'),
            (2, 4, 6, 'b'),
            (5, 10, 20, 'b'),
            (11, 15, 150, 'b'),
        ],
        columns=['y', 'x1', 'x2', 'group']
    )

    df_groups_lag_reg = pd.DataFrame(data=[
        (Timestamp('2000-01-01 00:00:00'), 1, 2, 3, nan, nan, 'a', nan, nan, nan, nan),
        (Timestamp('2000-01-02 00:00:00'), 4, 5, 8, 2.0, 3.0, 'a', 3.0, 5.0, nan, nan),
        (Timestamp('2000-01-03 00:00:00'), 10, 11, 100, 5.0, 8.0, 'a', 6.0, 92.0, 3.0, 5.0),
        (Timestamp('2000-01-04 00:00:00'), 15, 12, 40, 11.0, 100.0, 'a', 1.0, -60.0, 6.0, 92.0),
        (Timestamp('2000-01-06 00:00:00'), 22, 18, 82, 12.0, 40.0, 'a', 6.0, 42.0, 1.0, -60.0),
        (Timestamp('2000-01-07 00:00:00'), 46, 10, 61, 18.0, 82.0, 'a', -8.0, -21.0, 6.0, 42.0),
        (Timestamp('2000-01-01 00:00:00'), 2, 4, 6, nan, nan, 'b', nan, nan, nan, nan),
        (Timestamp('2000-01-02 00:00:00'), 5, 10, 20, 4.0, 6.0, 'b', 6.0, 14.0, nan, nan),
        (Timestamp('2000-01-03 00:00:00'), 11, 15, 150, 10.0, 20.0, 'b', 5.0, 130.0, 6.0, 14.0),
        (Timestamp('2000-01-04 00:00:00'), 13, 12, 156, 15.0, 150.0, 'b', -3.0, 6.0, 5.0, 130.0),
        (Timestamp('2000-01-07 00:00:00'), 13, 12, 156, nan, nan, 'b', nan, nan, nan, nan),
    ], columns=['Date', 'y', 'x1', 'x2', 'x1_lag_pregen',
                'x2_lag_pregen', 'group', 'x1_diff_pregen', 'x2_diff_pregen',
                'x1_diff_lag_pregen', 'x2_diff_lag_pregen'])

    yvar = 'y'
    xvars = ['x1', 'x2']
    all_xvars = ['const', 'x1', 'x2']
    groupvar = 'group'
    fe_xvars = all_xvars + ['b']

class RegTest(DataFrameTest):

    def _reg_test(self, df, expect_params, expect_cov, **reg_kwargs):
        reg_result = regtools.reg(df, self.yvar, self.xvars, **reg_kwargs)

        assert_series_equal(expect_params, reg_result.params)
        assert_frame_equal(expect_cov, reg_result.cov_params())

def compare_params_and_pvalues(result1, result2):
    assert (result1.params.values == result2.params.values).all()
    assert (result1.pvalues.values == result2.pvalues.values).all()


#### Actual test cases below ####

class TestReg(RegTest):

    def test_reg_simple(self):

        expect_params = pd.Series(data = [
            0.17948580753899121,
            0.31490944113004432,
            0.063565103400305134,
            ], index=self.all_xvars)

        expect_cov = pd.DataFrame(data = [
                (inf, inf, -inf),
                (inf, inf, -inf),
                (-inf, -inf, inf),
                ], columns=self.all_xvars, index=self.all_xvars)

        self._reg_test(self.df, expect_params, expect_cov, robust=False, cluster=False)

    def test_reg_nocons(self):
        expect_params = pd.Series(data = [
            0.41916167664670689,
            0.053892215568862201,
            ], index=self.xvars)

        expect_cov = pd.DataFrame(data = [
            (inf, -inf),
            (-inf, inf),
            ], columns =self.xvars, index=self.xvars)

        self._reg_test(self.df, expect_params, expect_cov, robust=False, cluster=False, cons=False)

    def test_reg_cluster(self):
        expect_params = pd.Series(data=[
            -0.057782704189492023,
            0.56915450458771888,
            0.023236241968922107,
        ], index=self.all_xvars)

        expect_cov = pd.DataFrame(data=[
            (1.3825827000648401, -0.64860156573107908, 0.037973583240341238),
            (-0.64860156573067984, 0.30427401633831397, -0.017814287380373151),
            (0.03797358324036447, -0.017814287380394575, 0.0010429705391543902),
        ], columns=self.all_xvars, index=self.all_xvars)

        self._reg_test(self.df_groups, expect_params, expect_cov, robust=False, cluster=[self.groupvar])

    def test_reg_fe(self):
        expect_params = pd.Series(data = [
            0.5140934,
            0.60590361,
            0.02298608,
            -1.71967829,
            ], index=self.fe_xvars)

        expect_cov = pd.DataFrame(data = [
            (0.86642167435809347, -0.15421903256428801, 0.0088046661162774695, 0.10782189494174726),
            (-0.15421903256428801, 0.043999645145278127, -0.0029669956299292195, -0.097047126884220028),
            (0.0088046661162774695, -0.0029669956299292195, 0.00024317047738642904, 0.0056102902997011801),
            (0.10782189494174726, -0.097047126884220028, 0.0056102902997011801, 0.76804342596454134),
        ], columns=self.fe_xvars, index=self.fe_xvars)

        self._reg_test(self.df_groups_no_nan, expect_params, expect_cov, robust=False, fe=self.groupvar)


class TestLagReg(DataFrameTest):

    def test_lag_reg(self):

        expected_result = regtools.reg(
            self.df_groups_lag_reg,
            'y',
            ['x1_lag_pregen', 'x2_lag_pregen']
        )

        lag_result = regtools.reg(
            self.df_groups_lag_reg,
            'y',
            ['x1', 'x2'],
            lag_variables=['x1', 'x2'],
            num_lags=1,
            lag_period_var='Date',
            lag_id_var='group',
            lag_fill_method=None
        )

        compare_params_and_pvalues(expected_result, lag_result)

class TestDiffReg(DataFrameTest):

    def test_diff_reg(self):
        expected_result = regtools.reg(
            self.df_groups_lag_reg,
            'y',
            ['x1_diff_pregen', 'x2_diff_pregen']
        )

        diff_result = regtools.chooser.any_reg(
            'diff',
            self.df_groups_lag_reg,
            'y',
            ['x1', 'x2'],
            diff_cols=['x1', 'x2'],
            difference_lag=1,
            date_col='Date',
            id_col='group',
            diff_fill_method=None
        )

        compare_params_and_pvalues(expected_result, diff_result)

    def test_diff_reg_lag(self):
        expected_result = regtools.reg(
            self.df_groups_lag_reg,
            'y',
            ['x1_diff_lag_pregen', 'x2_diff_lag_pregen']
        )

        diff_result = regtools.chooser.any_reg(
            'diff',
            self.df_groups_lag_reg,
            'y',
            ['x1', 'x2'],
            diff_cols=['x1', 'x2'],
            difference_lag=1,
            date_col='Date',
            id_col='group',
            lag_variables=['x1', 'x2'],
            num_lags=1,
            lag_period_var='Date',
            lag_id_var='group',
            diff_fill_method=None,
            lag_fill_method=None,
        )

        compare_params_and_pvalues(expected_result, diff_result)


class TestQuantReg(DataFrameTest):

    def test_quant_reg(self):
        result = regtools.chooser.any_reg(
            'quantile',
            self.df_groups_lag_reg,
            self.yvar,
            self.xvars,
            q=0.9
        )
        # TODO: check accuracy of quant reg result in test

    def test_quant_reg_iter(self):
        result = regtools.reg_for_each_xvar_set_and_produce_summary(
            self.df_groups_lag_reg,
            self.yvar,
            [self.xvars, self.xvars],
            reg_type='quantile',
            q=0.9
        )