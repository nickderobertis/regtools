

def _estimate_handling_robust_and_cluster(regdf, model, robust, cluster, **fit_kwargs):
    assert not (robust and cluster)  # need to pick one of robust or cluster

    if robust:
        return model.fit(cov_type='robust', **fit_kwargs)

    # TODO: linearmodels panel cluster
    if cluster:
        raise NotImplementedError('need to implement cluster for linearmodels panel models')

    return model.fit(**fit_kwargs)