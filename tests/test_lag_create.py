from regtools.lag.create import varname_to_lagged_varname


class TestVarnameToLaggedVarname:
    var_name = "my_var"
    lagged_var_name = "my_var$_{t - 1}$"

    def test_no_lags(self):
        lag_name = varname_to_lagged_varname(self.var_name, num_lags=0)
        assert lag_name == self.var_name

    def test_lag_on_unlagged(self):
        lag_name = varname_to_lagged_varname(self.var_name, num_lags=1)
        assert lag_name == self.lagged_var_name

    def test_lag_on_lagged(self):
        lag_name = varname_to_lagged_varname(self.lagged_var_name, num_lags=1)
        assert lag_name == "my_var$_{t - 2}$"
