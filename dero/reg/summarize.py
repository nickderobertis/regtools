from statsmodels.iolib.summary2 import summary_col


def produce_summary(reg_list, stderr=False, float_format='%0.1f'):

    summ =  summary_col(reg_list, stars=True, float_format=float_format,
                        info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                      'R2':lambda x: "{:.2f}".format(x.rsquared),
                      'Adj-R2':lambda x: "{:.2f}".format(x.rsquared_adj)})

    if not stderr:
        summ.tables[0].drop('', axis=0, inplace=True) #drops the rows containing standard errors

    return summ