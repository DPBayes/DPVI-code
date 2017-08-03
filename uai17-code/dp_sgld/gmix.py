from __future__ import division

import autograd.numpy as np
import autograd.numpy.random as npr
import sys 
from autograd.scipy.special import gammaln
from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import multivariate_normal, dirichlet
from autograd.core import primitive
from scipy.stats import invgamma


@primitive
def linvgamma(x,a,beta):
	return invgamma.logpdf(x,a,scale=beta)

linvgamma.defvjp(lambda g, ans, vs, gvs, x, a, scale: g * (-(a+1)/x + scale/x**2), argnum=0)

# Log likelihood of Gaussian mixture distribution
def logl(sample, k, value):
	pi = softmax(sample[:k])
	mvnpar = sample[k:]
	mus = mvnpar[:-k].reshape([k,int((mvnpar.shape[0]-k)/k)])
	taus = mvnpar[-k:]
	logps = [np.log(pi[i]) -0.5*(k*np.log(2*np.pi) + k*tau + np.dot((value-mus[i]), (value-mus[i]))/np.exp(tau))
                 for i, tau in enumerate(taus)]
	logps = np.array(logps)
	return logsumexp(logps)

# Log of prior density, assume std-mvnormal prior for mus and inversegamma prior for taus.
# Pi takes dirichlet prior
alpha = 1e1
beta = 1e-1

def pred_like(samples, test_data, k):
	# Method for prediction likelihood
	for sample in samples:
		logps = []
		tot_logl = 0
		for value in test_data:
			tot_logl += logl(sample, k, value)
		logps.append(tot_logl)
	return -np.log(len(samples)) + logsumexp(logps)



def logprior(sample, k):
	pi = softmax(sample[:k])
	a = np.ones(pi.shape[0])
	mvnpar = sample[k:]
	mus = mvnpar[:-k].reshape([k,int((mvnpar.shape[0]-k)/k)])
	taus = mvnpar[-k:]
	logp = 0
	for i, tau in enumerate(taus):
		logp += linvgamma(np.exp(tau), alpha, beta) + tau -k/2*np.log(2*np.pi) - 0.5*np.dot(mus[i], mus[i])
	return logp + dirichlet.logpdf(pi, a) + ldt(pi)

def ldt(x):
	return np.linalg.slogdet(np.diag(x+1e-9)-np.outer(x,x))[1]

def softmax(x, axis=None):
	x = (x.T - np.max(x, axis)).T  
	tmp = (np.exp(x).T / np.sum(np.exp(x), axis)).T
	tmp = ((tmp+1e-9).T / np.sum(tmp+1e-9, axis)).T
	return tmp



