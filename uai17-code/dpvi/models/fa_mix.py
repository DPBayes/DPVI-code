from __future__ import division

import autograd.numpy as np
from autograd.core import primitive
from autograd.scipy.special import gammaln
from autograd.scipy.stats import multivariate_normal, dirichlet
from autograd.scipy.misc import logsumexp
from utils.misc import softmax, softplus, stick_forward, stick_backward, stick_jacobian_det
from utils.advi import draw_samples
from scipy.stats import gamma


@primitive
def lgamma(x,alpha,beta):
	return gamma.logpdf(x,alpha,scale=beta)
lgamma.defvjp(lambda g, ans, vs, gvs, x, a, scale: g * ((a-1)/x - 1/scale), argnum=0)


def get_pars(sample, k, d, q):
	pars = {}
	par_names = ['pi', 'mu', 'alpha', 'tau', 'W']
	#par_dims = [1, d, q, 1, d*q]
	par_dims = [(k-1)/k, d, q, 1, d*q]
	indx0 = 0
	for i, name in enumerate(par_names):
		indx1 = int(indx0 + k*par_dims[i])
		pars[name]=sample[indx0: indx1]
		indx0 = indx1
	return pars

#def get_inv(W, tau):
	#if W.ndim>1:
		#d, q = W.shape
	#else: d = W.shape[0]; q = 1
	#D_minus = np.eye(d)*tau
	#tmp = np.dot(D_minus, W)
	#if q == 1:
		#tmp2 = 1/(1+np.dot(tmp.T, W))
		#return D_minus - tmp2*np.outer(tmp, tmp.T)

#def get_logdet(W, tau):
	#if W.ndim>1:
		#d, q = W.shape
	#else: 
		#d = W.shape[0]
		#q=1
	#tmp1 = -d*np.log(tau)
	#if q==1:
		#tmp2 = get_inv(W, tau)
		#tmp3 = np.log(1 - np.dot(np.dot(W.T, tmp2), W))
		#return tmp1-tmp3

def multi_lpdf(value, mean, cov):
	d = mean.shape[0]
	tmp2 = value-mean
	if value.ndim>1:
		return -0.5*(d*np.log(2*np.pi) + np.linalg.slogdet(cov)[1] + np.diag(np.dot(tmp2, np.dot(np.linalg.inv(cov),tmp2.T))))
	else:
		return -0.5*(d*np.log(2*np.pi) + np.linalg.slogdet(cov)[1] + np.dot(tmp2, np.dot(np.linalg.inv(cov),tmp2.T)))

#def multi_lpdf2(value, mean, W, tau):
	#d = mean.shape[0]
	#tmp2 = value-mean
	#if value.ndim>1:
		#return -0.5*(d*np.log(2*np.pi) + get_logdet(W, tau) + np.diag(np.dot(tmp2, np.dot(get_inv(W, tau), tmp2.T))))
	#else:
		#return -0.5*(d*np.log(2*np.pi) + get_logdet(W, tau) + np.dot(tmp2, np.dot(get_inv(W, tau), tmp2.T)))

def log_like(var_par, draw, data, k):
	if data.ndim>1:
		d = data.shape[1]
	else: d = data.shape[0]
	l = int(len(var_par)/2)
	q = ((l+1)/k-2-d)/(d+1)
	q = int(q)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	pars = get_pars(samples, k, d, q)
	
	#pi = softmax(pars['pi'])
	pi = stick_backward(pars['pi'])
	#pi = stick_backward(samples[:1])
	#pi = softmax(samples[:k])
	
	mus = pars['mu'].reshape([k,d])
	#mus = samples[k:k+k*d].reshape([k,d])
	#mus = samples[1:k*d+1].reshape([k,d])

	alphas = pars['alpha'].reshape([k,q])
	#alphas = samples[k+k*d:k+k*d+k*q].reshape([k,q])
	#alphas = samples[k*d+1:k*d+1+k*q].reshape([k,q])
	
	tau = np.exp(pars['tau'])
	#tau = np.exp(samples[k+k*d+k*q:k+k*d+k*q+k])
	#tau = np.exp(samples[k*d+1+k*q:k*d+1+k*q+k])
	
	W = pars['W'].reshape([k,d*q])
	#W = samples[k+k*d+k*q+k:].reshape([k,d*q])
	#W = samples[k*d+1+k*q+k:].reshape([k,d*q])
	
	logps = []
	for i, W_ in enumerate(W):
		W_ = W_.reshape([d,q])
		tau_ = tau[i]
		if q==1:
			tmp = np.outer(W_,W_.T)+np.eye(d)/tau_
		else:
			tmp = np.dot(W_,W_.T)+np.eye(d)/tau_
		logps.append(np.log(pi[i]) + multi_lpdf(data, mus[i], tmp))
		#logps.append(np.log(pi[i]) + multi_lpdf2(data, mus[i], W_, tau_))
	logps = np.array(logps)
	return np.sum(logsumexp(logps, axis = 0))

def logprior(var_par, draw, k, a=None, b=None, d=None):
	# Log of prior probabilities
	l = int(len(var_par)/2)
	q = ((l+1)/k-2-d)/(d+1)
	q = int(q)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	pars = get_pars(samples, k, d, q)

	#pi = softmax(pars['pi'])
	pi = stick_backward(pars['pi'])

	mus = pars['mu'].reshape([k,d])

	alphas = pars['alpha'].reshape([k,q])

	tau = pars['tau']

	W = pars['W'].reshape([k,d*q])

	logp = 0
	for j, W_ in enumerate(W):
		W_ = W_.reshape([d,q])
		tau_ = tau[j]
		logp += multi_lpdf(mus[j], mean = np.zeros(d), cov=1e3*np.eye(d)) + sum([multi_lpdf(W_[:,i], mean = np.zeros(d), cov=np.eye(d)/alpha) 
														for i, alpha in enumerate(np.exp(alphas[j]))]) + sum(lgamma(np.exp(alphas[j]), 1e-3, 1e3) + alphas[j]) + lgamma(np.exp(tau_), 1e-3, 1e3) + tau_
	#return logp + dirichlet.logpdf(pi, np.ones(k)) + ldt(pi)
	return logp + dirichlet.logpdf(pi, np.ones(k)) + np.log(np.abs(stick_jacobian_det(pars['pi'])))

def ldt(x):
	return np.linalg.slogdet(np.diag(x+1e-15)-np.outer(x,x))[1]


def pred_like(data, var_par, samples, k):
	plike = 0
	for y in data:
		logpp = []
		for s in samples:
			logpp.append(log_like(var_par, s, y, k))
		plike += logsumexp(logpp)-np.log(samples.shape[0])
	return plike/data.shape[0]









