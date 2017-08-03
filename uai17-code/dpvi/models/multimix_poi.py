from __future__ import division

import autograd.numpy as np
from autograd.core import primitive
from autograd.scipy.special import gammaln
from autograd.scipy.stats import multivariate_normal, dirichlet
from autograd.scipy.misc import logsumexp
from utils.misc import softmax, softplus
from utils.advi import draw_samples
from scipy.stats import invgamma

@primitive
def linvgamma(x,a,beta):
	return invgamma.logpdf(x,a,scale=beta)
linvgamma.defvjp(lambda g, ans, vs, gvs, x, a, scale: g * (-(a+1)/x + scale/x**2), argnum=0)


# Multinomial mixture model
def log_like(var_par, draw, value, k):
	d = len(value)
	value = value[:-1]
	l = int(len(var_par)/2)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	pi = softmax(samples[:k])
	lam = np.exp(samples[k:2*k])
	thetas = softmax(samples[2*k:].reshape([k,d]), axis=1)
	thetas1 = thetas[:,:-1]
	thetas2 = thetas[:,-1]
	logps = np.log(pi) + np.dot(np.log(thetas1.T*lam).T, value) + lam*(thetas2-1) - np.sum(gammaln(value+1))  
	return logsumexp(logps)

def np_log_like(var_par, draw, data, k):
	d = data.shape[1]
	gln_test_values = np.sum(gammaln(data[:,:-1]+1), axis=1)
	data = data[:,:-1]
	l = int(len(var_par)/2)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	pi = softmax(samples[:k])
	lam = np.exp(samples[k:2*k])
	thetas = softmax(samples[2*k:].reshape([k,d]), axis=1)
	thetas1 = thetas[:,:-1]
	thetas2 = thetas[:,-1]
	logps = (np.log(pi) + np.dot(data, np.log(thetas1.T*lam)) + lam*(thetas2-1)).T  - gln_test_values
	return np.sum(logsumexp(logps, axis=0))


def logprior(var_par, draw, k, a=None, b=None):
	# Log of prior probabilities
	d = int((len(var_par)/2-2*k)/k)
	l = int(len(var_par)/2)
	alpha = np.ones(k) # Prior for mixture probabilities
	beta = np.ones(d) # Prior for multinomials

	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	pi = softmax(samples[:k])
	lam = samples[k:2*k]
	thetas = softmax(samples[2*k:].reshape([k,d]), axis=1)
	return dirichlet.logpdf(pi, alpha) + np.sum([dirichlet.logpdf(theta, beta) + ldt(theta) for theta in thetas]) + ldt(pi) +\
		sum(linvgamma(np.exp(lam), a, b) + lam)


def ldt(x):
	return np.linalg.slogdet(np.diag(x+1e-9)-np.outer(x,x))[1]


from scipy.special import gammaln
from scipy.misc import logsumexp as logsumexp2


def pred_like(test_data, var_par, n_samples, k):
	# Method for prediction likelihood
	N_test, d = test_data.shape
	l = int(len(var_par)/2)
	gln_test_values = np.sum(gammaln(test_data[:,:-1]+1), axis=1)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	like_matrix = np.empty([N_test, n_samples])
	samples = draw_samples(var_par, n_samples)
	for s, sample in enumerate(samples):
		pi = softmax(sample[:k])
		lam = np.exp(sample[k:2*k])
		thetas = softmax(sample[2*k:].reshape([k,d]), axis=1)
		thetas1 = thetas[:,:-1]
		thetas2 = thetas[:,-1]
		logps = (np.log(pi) + np.dot(test_data[:,:-1], np.log(thetas1.T*lam)) + lam*(thetas2-1)).T  - gln_test_values
		like_matrix[:,s] = logsumexp2(np.stack(logps)[:, :N_test], axis=0)
	return np.mean(logsumexp2(like_matrix, axis=1)-np.log(n_samples))
