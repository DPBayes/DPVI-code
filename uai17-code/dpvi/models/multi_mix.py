from __future__ import division

import autograd.numpy as np
from autograd.core import primitive
from autograd.scipy.special import gammaln
from autograd.scipy.stats import multivariate_normal, dirichlet
from autograd.scipy.misc import logsumexp
from utils.misc import softmax
from utils.advi import draw_samples

# Multinomial mixture model
def log_like(var_par, draw, value, k):
	l = int(len(var_par)/2)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	pi = softmax(samples[:k])
	d = len(value)
	thetas = softmax(samples[k:].reshape([k,d]), axis=1)
	n = np.sum(value)
	logps = np.log(pi) + np.dot(np.log(thetas), value) + gammaln(n+1) - np.sum(gammaln(value+1))  
	return logsumexp(logps)

def logprior(var_par, draw, k):
	# Log of prior probabilities
	d = int((len(var_par)/2-k)/k)
	l = int(len(var_par)/2)
	alpha = np.ones(k) # Prior for mixture probabilities
	beta = np.ones(d) # Prior for multinomials
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	pi = softmax(samples[:k])
	thetas = softmax(samples[k:].reshape([k,d]), axis=1)
	return dirichlet.logpdf(pi, alpha)+np.sum([dirichlet.logpdf(theta, beta) + ldt(theta) for theta in thetas]) + ldt(pi)

def ldt(x):
	return np.linalg.slogdet(np.diag(x+1e-9)-np.outer(x,x))[1]


from scipy.special import gammaln
from scipy.misc import logsumexp as logsumexp2


def pred_like(test_data, var_par, n_samples, k):
	# Method for prediction likelihood
	N_test, d = test_data.shape
	l = int(len(var_par)/2)
	gln_test_n = gammaln(test_data.sum(axis=1)+1)
	gln_test_values = np.sum(gammaln(test_data+1), axis=1)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	like_matrix = np.empty([N_test, n_samples])
	samples = draw_samples(var_par, n_samples)
	for s, sample in enumerate(samples):
		pi = softmax(sample[:k])
		thetas = softmax(sample[k:].reshape([k,d]), axis=1)
		logps = (np.log(pi) + np.dot(test_data, np.log(thetas.T))).T + gln_test_n -gln_test_values
		like_matrix[:,s] = logsumexp2(np.stack(logps)[:, :N_test], axis=0)
	return np.mean(logsumexp2(like_matrix, axis=1)-np.log(n_samples))






#@primitive
#def logsumexp(x):
    #"""Numerically stable log(sum(exp(x)))"""
    #max_x = np.max(x)
    #return max_x + np.log(np.sum(np.exp(x - max_x)))
#def logsumexp_vjp(g, ans, vs, gvs, x):
    #return np.full(x.shape, g) * np.exp(x - np.full(x.shape, ans))
#logsumexp.defvjp(logsumexp_vjp)


#def log_like2(var_par, draw, value, k):
	#l = int(len(var_par)/2)
	#mu, cov = var_par[:l], np.exp(var_par[l:])
	#samples = draw*cov + mu
	#pi = softmax(samples[:k])
	#d = value.shape[1]
	#thetas = softmax(samples[k:].reshape([k,d]), axis=1)
	#n = np.sum(value, axis=1)
	#logps = (np.log(pi) + np.dot(value, np.log(thetas).T) ).T + gammaln(n+1) - np.sum(gammaln(value+1), axis=1)  
	#return logsumexp2(logps.T, axis=1)

