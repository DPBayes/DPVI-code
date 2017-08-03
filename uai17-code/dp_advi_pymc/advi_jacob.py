
'''
(c) 2016, John Salvatier & Taku Yoshioka
'''
from __future__ import division
from __future__ import print_function
from collections import OrderedDict, namedtuple

import numpy as np
import theano
import theano.tensor as tt
#from lasagne.updates import norm_constraint
from theano.sandbox.rng_mrg import MRG_RandomStreams

import pymc3 as pm
from pymc3.backends.base import MultiTrace

from tqdm import trange

__all__ = ['advi', 'sample_vp']

ADVIFit = namedtuple('ADVIFit', 'means, stds, elbo_vals')


def check_discrete_rvs(vars):
    """Check that vars not include discrete variables, excepting ObservedRVs.
    """
    vars_ = [var for var in vars if not isinstance(var, pm.model.ObservedRV)]
    if any([var.dtype in pm.discrete_types for var in vars_]):
        raise ValueError('Model should not include discrete RVs for ADVI.')


def gen_random_state():
    """Helper to generate a random state for MRG_RandomStreams"""
    M1 = 2147483647
    M2 = 2147462579
    return np.random.randint(0, M1, 3).tolist() + np.random.randint(0, M2, 3).tolist()


def advi(vars=None, start=None, model=None, n=5000, accurate_elbo=False,
         optimizer=None, learning_rate=.001, epsilon=.1, random_seed=None,
         verbose=1, dp_par=None):
    """Perform automatic differentiation variational inference (ADVI).
    This function implements the meanfield ADVI, where the variational
    posterior distribution is assumed to be spherical Gaussian without
    correlation of parameters and fit to the true posterior distribution.
    The means and standard deviations of the variational posterior are referred
    to as variational parameters.
    The return value of this function is an :code:`ADVIfit` object, which has
    variational parameters. If you want to draw samples from the variational
    posterior, you need to pass the :code:`ADVIfit` object to
    :code:`pymc3.variational.sample_vp()`.
    The variational parameters are defined on the transformed space, which is
    required to do ADVI on an unconstrained parameter space as described in
    [KTR+2016]. The parameters in the :code:`ADVIfit` object are in the
    transformed space, while traces returned by :code:`sample_vp()` are in
    the original space as obtained by MCMC sampling methods in PyMC3.
    The variational parameters are optimized with given optimizer, which is a
    function that returns a dictionary of parameter updates as provided to
    Theano function. If no optimizer is provided, optimization is performed
    with a modified version of adagrad, where only the last (n_window) gradient
    vectors are used to control the learning rate and older gradient vectors
    are ignored. n_window denotes the size of time window and fixed to 10.
    Parameters
    ----------
    vars : object
        Random variables.
    start : Dict or None
        Initial values of parameters (variational means).
    model : Model
        Probabilistic model.
    n : int
        Number of interations updating parameters.
    accurate_elbo : bool
        If true, 100 MC samples are used for accurate calculation of ELBO.
    optimizer : (loss, tensor) -> dict or OrderedDict
        A function that returns parameter updates given loss and parameter
        tensor. If :code:`None` (default), a default Adagrad optimizer is
        used with parameters :code:`learning_rate` and :code:`epsilon` below.
    learning_rate: float
        Base learning rate for adagrad. This parameter is ignored when
        optimizer is given.
    epsilon : float
        Offset in denominator of the scale of learning rate in Adagrad.
        This parameter is ignored when optimizer is given.
    random_seed : int or None
        Seed to initialize random state. None uses current seed.
    Returns
    -------
    ADVIFit
        Named tuple, which includes 'means', 'stds', and 'elbo_vals'.
    'means' is the mean. 'stds' is the standard deviation.
    'elbo_vals' is the trace of ELBO values during optimizaiton.
    References
    ----------
    .. [KTR+2016] Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A.,
        and Blei, D. M. (2016). Automatic Differentiation Variational Inference.
        arXiv preprint arXiv:1603.00788.
    """
    model = pm.modelcontext(model)
    if start is None:
        start = model.test_point

    if vars is None:
        vars = model.vars
    vars = pm.inputvars(vars)

    check_discrete_rvs(vars)

    n_mcsamples = 100 if accurate_elbo else 1

    # Prepare optimizer
    if optimizer is None:
        optimizer = adagrad_optimizer(learning_rate, epsilon)

    # Create variational gradient tensor
    elbo, shared = _calc_elbo(vars, model, n_mcsamples=n_mcsamples,
                              random_seed=random_seed)

    # Set starting values
    for var, share in shared.items():
        share.set_value(start[str(var)])

    order = pm.ArrayOrdering(vars)
    bij = pm.DictToArrayBijection(order, start)
    u_start = bij.map(start)
    w_start = np.zeros_like(u_start)
    uw = np.concatenate([u_start, w_start])

    # Create parameter update function used in the training loop
    uw_shared = theano.shared(uw, 'uw_shared')
    elbo = pm.CallableTensor(elbo)(uw_shared)
    updates = optimizer(likeloss=-1 * elbo[0], entroloss =-1 * elbo[1], param=uw_shared, dp_par=dp_par, n_par = len(vars))
    f = theano.function([], [uw_shared, tt.add(elbo[1],tt.sum(elbo[0], axis=0))], updates=updates)

    # Optimization loop
    elbos = np.empty(n)
    try:
        for i in range(n):
            uw_i, e = f()
            elbos[i] = e
            if verbose and not i % (n // 10):
                if not i:
                    print('Iteration {0} [{1}%]: ELBO = {2}'.format(
                        i, 100 * i // n, e.round(2)))
                else:
                    avg_elbo = elbos[i - n // 10:i].mean()
                    print('Iteration {0} [{1}%]: Average ELBO = {2}'.format(
                        i, 100 * i // n, avg_elbo.round(2)))
    except KeyboardInterrupt:
        if verbose:
            elbos = elbos[:i]
            avg_elbo = elbos[i - n // 10:].mean()
            print('Interrupted at {0} [{1}%]: Average ELBO = {2}'.format(
                i, 100 * i // n, avg_elbo.round(2)))
    else:
        if verbose:
            avg_elbo = elbos[-n // 10:].mean()
            print('Finished [100%]: Average ELBO = {}'.format(avg_elbo.round(2)))

    # Estimated parameters
    l = int(uw_i.size / 2)
    u = bij.rmap(uw_i[:l])
    w = bij.rmap(uw_i[l:])
    # w is in log space
    for var in w.keys():
        w[var] = np.exp(w[var])

    return ADVIFit(u, w, elbos)


def _calc_elbo(vars, model, n_mcsamples, random_seed):
    """Calculate approximate ELBO.
    """
    theano.config.compute_test_value = 'ignore'
    shared = pm.make_shared_replacements(vars, model)

    factors = [var.logp_elemwiset for var in model.basic_RVs] + model.potentials
    wfactors = tt.stack(factors[:-1])
    logpt = tt.concatenate([wfactors, factors[-1].reshape([factors[-1].size])])
    
    #wfactors = tt.stack(factors[:-1])
    #logpt = tt.concatenate([wfactors, factors[-1]])
    
    #logpt = tt.add(*map(tt.sum, factors))

    [logp], inarray = pm.join_nonshared_inputs([logpt], vars, shared)

    uw = tt.dvector('uw')
    uw.tag.test_value = np.concatenate([inarray.tag.test_value,
                                        inarray.tag.test_value])

    elbo = _elbo_t(logp, uw, inarray, n_mcsamples, random_seed)

    return elbo, shared


def _elbo_t(logp, uw, inarray, n_mcsamples, random_seed):
    """Create Theano tensor of approximate ELBO by Monte Carlo sampling.
    """
    l = (uw.size / 2).astype('int64')
    u = uw[:l]
    w = uw[l:]

    # Callable tensor
    logp_ = lambda input: theano.clone(logp, {inarray: input}, strict=False)

    # Naive Monte-Carlo
    if random_seed is None:
        r = MRG_RandomStreams(gen_random_state())
    else:
        r = MRG_RandomStreams(seed=random_seed)

    if n_mcsamples == 1:
        n = r.normal(size=[u.tag.test_value.shape[0]])
        q = n * tt.exp(w) + u
        elbo = [logp_(q), tt.sum(w) + 0.5 * l * (1 + np.log(2.0 * np.pi))]
    else:
        n = r.normal(size=(n_mcsamples, u.tag.test_value.shape[0]))
        qs = n * tt.exp(w) + u
        logps, _ = theano.scan(fn=lambda q: logp_(q),
                               outputs_info=None,
                               sequences=[qs])
        elbo = tt.mean(logps) + tt.sum(w) + 0.5 * l * (1 + np.log(2.0 * np.pi))

    return elbo 


def adagrad_optimizer(learning_rate, epsilon, n_win=10):
    """Returns a function that returns parameter updates.
    Parameter
    ---------
    learning_rate : float
        Learning rate.
    epsilon : float
        Offset to avoid zero-division in the normalizer of adagrad.
    n_win : int
        Number of past steps to calculate scales of parameter gradients.
    Returns
    -------
    A function (loss, param) -> updates.
    loss : Theano scalar
        Loss function to be minimized (e.g., negative ELBO).
    param : Theano tensor
        Parameters to be optimized.
    updates : OrderedDict
        Parameter updates used in Theano functions.
    """
    def optimizer(likeloss, entroloss , param, dp_par, n_par):
        r = MRG_RandomStreams(gen_random_state())
        if dp_par is not None: 
            [sigma, C] = dp_par
        i = theano.shared(np.array(0))
        value = param.get_value(borrow=True)
        d = n_par
        n = likeloss.shape[0]-d
        accu = theano.shared(
            np.zeros(value.shape + (n_win,), dtype=value.dtype))
        likeloss = tt.add(likeloss[d:], tt.sum(likeloss[:d])/n)
        J = theano.scan(lambda i, likeloss, param: tt.grad(likeloss[i], param), sequences=tt.arange(n), non_sequences=[likeloss,param])[0]
        tmp = tt.grad(entroloss, param)/n
        ttmp = J+tmp

        if dp_par is None: grad = tt.sum(ttmp, axis=0)
        else:
            c_gnorm = tt.clip(ttmp.norm(2, axis=1), 0, C)
            c_grad = (ttmp.T/ttmp.norm(2, axis=1))*c_gnorm
            
            grad = tt.sum(c_grad.T, axis=0) + r.normal(size=param.shape, std = 2*sigma*C)
            #grad = norm_constraint(ttmp, C, norm_axes=(1,)).sum(axis=0) + r.normal(size=param.shape, std = 2*sigma*C)

        # Append squared gradient vector to accu_new
        accu_new = tt.set_subtensor(accu[:, i], grad ** 2)
        i_new = tt.switch((i + 1) < n_win, i + 1, 0)

        updates = OrderedDict()
        updates[accu] = accu_new
        updates[i] = i_new

        accu_sum = accu_new.sum(axis=1)
        updates[param] = param - (learning_rate * grad /
                                  tt.sqrt(accu_sum + epsilon))

        return updates
    return optimizer


def sample_vp(
        vparams, draws=1000, model=None, local_RVs=None, random_seed=None,
        hide_transformed=True, progressbar=True):
    """Draw samples from variational posterior.
    Parameters
    ----------
    vparams : dict or pymc3.variational.ADVIFit
        Estimated variational parameters of the model.
    draws : int
        Number of random samples.
    model : pymc3.Model
        Probabilistic model.
    random_seed : int or None
        Seed of random number generator.  None to use current seed.
    hide_transformed : bool
        If False, transformed variables are also sampled. Default is True.
    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        Samples drawn from the variational posterior.
    """
    model = pm.modelcontext(model)

    if isinstance(vparams, ADVIFit):
        vparams = {
            'means': vparams.means,
            'stds': vparams.stds
        }

    ds = model.deterministics
    get_transformed = lambda v: v if v not in ds else v.transformed
    rvs = lambda x: [get_transformed(v) for v in x] if x is not None else []

    global_RVs = list(set(model.free_RVs) - set(rvs(local_RVs)))

    # Make dict for replacements of random variables
    if random_seed is None:
        r = MRG_RandomStreams(seed=123)
    else:
        r = MRG_RandomStreams(seed=123)
    updates = {}
    for v in global_RVs:
        u = theano.shared(vparams['means'][str(v)]).ravel()
        w = theano.shared(vparams['stds'][str(v)]).ravel()
        n = r.normal(size=u.tag.test_value.shape)
        updates.update({v: (n * w + u).reshape(v.tag.test_value.shape)})

    if local_RVs is not None:
        ds = model.deterministics
        get_transformed = lambda v: v if v not in ds else v.transformed
        for v_, (uw, _) in local_RVs.items():
            v = get_transformed(v_)
            u = uw[0].ravel()
            w = uw[1].ravel()
            n = r.normal(size=u.tag.test_value.shape)
            updates.update(
                {v: (n * tt.exp(w) + u).reshape(v.tag.test_value.shape)})

    # Replace some nodes of the graph with variational distributions
    vars = model.free_RVs
    samples = theano.clone(vars, updates)
    f = theano.function([], samples)

    # Random variables which will be sampled
    vars_sampled = [v for v in model.unobserved_RVs if not str(v).endswith('_')] \
        if hide_transformed else \
                   [v for v in model.unobserved_RVs]

    varnames = [str(var) for var in model.unobserved_RVs]
    trace = pm.sampling.NDArray(model=model, vars=vars_sampled)
    trace.setup(draws=draws, chain=0)

    range_ = trange(draws) if progressbar else range(draws)

    for i in range_:
        # 'point' is like {'var1': np.array(0.1), 'var2': np.array(0.2), ...}
        point = {varname: value for varname, value in zip(varnames, f())}
        trace.record(point)

	return MultiTrace([trace])