
class RegressionSetArgs:
    def __init__(self, df, yvar, xvars_list, fe_list=None, **reg_kwargs):
        self.df = df
        self.yvar = yvar
        self.xvars_list = xvars_list
        self.fe = fe_list
        self._parse_reg_kwargs(reg_kwargs)

    def keys(self):
        return ['df', 'yvar', 'xvars_list', 'fe'] + self.reg_kwarg_names

    def _parse_reg_kwargs(self, reg_kwargs):
        self.reg_kwarg_names = []

        for name, arg in reg_kwargs.items():
            # for each arg in reg_kwargs, add as class attribute
            setattr(self, name, arg)
            self.reg_kwarg_names.append(name) #also add to list of names for keys()

    def __getitem__(self, item):
        return getattr(self, item)