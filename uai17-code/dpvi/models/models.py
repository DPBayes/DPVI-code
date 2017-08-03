from __future__ import division

import autograd.numpy as np
from autograd.core import primitive
from autograd.scipy.special import gammaln
from scipy.stats import gamma

# Densities in autograd
from autograd.scipy.stats import multivariate_normal, dirichlet, norm, t


def get_rvs(name):
	if(name == 'Normal'): rv = norm
	elif(name == 'MvNormal'): rv = multivariate_normal
	elif(name == 'Dirichlet'): rv = dirichlet
	return rv

class Model():
	
	def __init__(self, likelihood, priors, observed, start):
		self.likelihood = likelihood
		self.prior = priors
		self.observed = observed
		self.start = start
	
	def built_joint_log_prob(self):
		rvs = {}
		params= {}
		rv_list = self.prior
		for rv in rv_list:
			name, params_ = rv.split('(')
			params_ = params_[:-1].split(',')
			rvs[name] = get_rvs(name)
			params[name] = [float(p) for p in params_]
		like_rv = self.likelihood
		like_name, like_params = like_rv.split('(')
		like_params = like_params[:-1].split(',')
		like_rv = get_rvs(like_name)
		self.logp = np.sum(like_rv.logpdf(self.observed, float(like_params[0]), float(like_params[1]))) + np.sum([rvs[rv].logpdf(self.start[rv], params[rv][0], params[rv][1]) for rv in rvs])