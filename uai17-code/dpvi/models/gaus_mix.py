from __future__ import division

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import multivariate_normal, dirichlet
from autograd.core import primitive
from scipy.stats import invgamma
from utils.misc import softmax, softplus, stick_forward, stick_backward, stick_jacobian_det

npr.seed(123)


@primitive
def linvgamma(x,a,beta):
	return invgamma.logpdf(x,a,scale=beta)

linvgamma.defvjp(lambda g, ans, vs, gvs, x, a, scale: g * (-(a+1)/x + scale/x**2), argnum=0)

def get_pars(sample, k, d):
	pars = {}
	par_names = ['pi', 'mu', 'tau']
	par_dims = [(k-1)/k, d, 1]
	indx0 = 0
	for i, name in enumerate(par_names):
		indx1 = int(indx0 + k*par_dims[i])
		pars[name]=sample[indx0: indx1]
		indx0 = indx1
	return pars

def multi_lpdf(value, mean, tau):
	d = mean.shape[0]
	tmp2 = value-mean
	if value.ndim>1:
		return -0.5*(d*np.log(2*np.pi) + d*np.log(tau) + np.diag(np.dot(tmp2, np.dot(np.eye(d)/tau,tmp2.T))))
	else:
		return -0.5*(d*np.log(2*np.pi) + d*np.log(tau) + np.dot(tmp2, tmp2)/tau)


# Log likelihood of Gaussian mixture distribution
def logl(var_par, draw, value, k):
	l = int(len(var_par)/2)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	d = value.T.shape[0]
	pars = get_pars(samples, k, d)
	pi = stick_backward(pars['pi'])
	mus = pars['mu'].reshape([k,d])
	taus = np.exp(pars['tau'])
	logps = [np.log(pi[i]) + multi_lpdf(value, mus[i], tau) for i, tau in enumerate(taus)]
	logps = np.array(logps)
	return np.sum(logsumexp(logps, axis=0))

# Log of prior density, assume std-mvnormal prior for mus and inversegamma prior for taus.
# Pi takes dirichlet prior
alpha = 1e1
beta = 1e-1

def logprior(var_par, draw, k):
	# Log of prior probabilities
	l = int(len(var_par)/2)
	d = (l-2*k+1)/k
	d = int(d)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	pars = get_pars(samples, k, d)
	pi = stick_backward(pars['pi'])
	mus = pars['mu'].reshape([k,d])
	taus = pars['tau']
	logp = 0
	for i, tau in enumerate(taus):
		logp += linvgamma(np.exp(tau), alpha, beta) + tau + multi_lpdf(mus[i], np.zeros(d), 1)
	return logp + dirichlet.logpdf(pi, np.ones(k)) + np.log(np.abs(stick_jacobian_det(pars['pi'])))




#def logprior(var_par, draw, k):
	#sample = var_par[:int(len(var_par)/2)] + np.exp(var_par[int(len(var_par)/2):])*draw
	#pi = softmax(sample[:k])
	#a = np.ones(pi.shape[0])
	#mvnpar = sample[k:]
	#mus = mvnpar[:-k].reshape([k,int((mvnpar.shape[0]-k)/k)])
	#taus = mvnpar[-k:]
	#logp = 0
	#for i, tau in enumerate(taus):
		#logp += linvgamma(np.exp(tau), alpha, beta) + tau -k/2*(np.log(2*np.pi) + 2*np.log(tau0)) - 0.5*np.dot(mus[i], mus[i])/tau0**2
	#return logp + dirichlet.logpdf(pi, a) + ldt(pi)

#def ldt(x):
	#return np.linalg.slogdet(np.diag(x+1e-9)-np.outer(x,x))[1]

#def softmax(x, axis=None):
	#x = (x.T - np.max(x, axis)).T  
	#tmp = (np.exp(x).T / np.sum(np.exp(x), axis)).T
	#tmp = ((tmp+1e-9).T / np.sum(tmp+1e-9, axis)).T
	#return tmp



def pred_like(var_par, test_data, k, n_mc=10):
	# Method for prediction likelihood
	l = int(len(var_par)/2)
	n_test,d = test_data.shape
	mu, cov = var_par[:l], np.exp(var_par[l:])
	for s in range(0,n_mc):
		draw = npr.randn(l)
		logps = []
		tot_logl = 0
		for value in test_data:
			tot_logl += logl(var_par, draw, value, k)
		logps.append(tot_logl)
	return -np.log(n_mc) + logsumexp(logps)

