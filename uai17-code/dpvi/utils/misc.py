from __future__ import division
import autograd.numpy as np
import numpy as onp
from autograd.numpy import numpy_wrapper as anp

from ma import gaussian_moments as gm

from autograd.numpy.numpy_grads import grad_np_cumsum
def cumprod(x):
	cp = []
	for i in range(len(x)):
		cp.append(anp.prod(x[:i+1]))
	return np.array(cp)

def softmax(x, axis=None):
	return (np.exp(x).T / np.sum(np.exp(x), axis)).T

def softplus(x):
	return np.log(1+np.exp(x))

def act(sigma, delta_tot, delta_prime, T, q):
	d_iter = (delta_tot-delta_prime)/(T*q)
	sigma_prime = np.log(1+q*(np.exp(np.sqrt(2*np.log(1.25/d_iter))/sigma)-1))
	return np.sqrt(2*T*np.log(1.25/delta_prime))*sigma_prime+T*sigma_prime*(np.exp(sigma_prime)-1)

def ma(sigma, delta, T, q, max_moment=150):
	lm = []
	for i in range(1,max_moment):
		tmp = gm.compute_log_moment(q, sigma, T, i)
		lm += [(i, tmp)]
	return gm._compute_eps(lm, delta) 

from scipy.optimize import minimize

def find_sigma(epsilon, delta_tot, T, q):
	def fun(sigma):
	    return np.abs(2*ma(2*sigma, 0.5*delta_tot, T, q, max_moment=120)-epsilon)
	tmp = minimize(fun, 1.0, method='Nelder-Mead', tol = 1e-3,  options={'maxiter':30})
	return tmp['x'][0], tmp['fun']

def find_sigma_act(epsilon, delta_tot, T, q):
	def fun(sigma):
	    return np.abs(2*act(2*sigma, 0.5*delta_tot, 0.1*delta_tot, T, q)-epsilon)
	tmp = minimize(fun, 1.0, method='Nelder-Mead', tol = 1e-5,  options={'maxiter':100})
	return tmp['x'][0], tmp['fun']


def logit(x):
	return np.log(x) - np.log(1-x)

def invlogit(y):
	return np.exp(y)/(np.exp(y)+1)

def stick_forward(x_):
	x = x_.T
	# reverse cumsum
	x0 = x[:-1]
	s = np.cumsum(x0[::-1], 0)[::-1] + x[-1]
	z = x0 / s
	Km1 = x.shape[0] - 1
	k = np.arange(Km1)[(slice(None), ) + (None, ) * (x.ndim - 1)]
	eq_share = logit(1. / (Km1 + 1 - k))  # - np.log(Km1 - k)
	y = logit(z) - eq_share
	return y.T

def stick_backward(y_):
	y = y_.T
	Km1 = y.shape[0]
	k = np.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
	eq_share = logit(1. / (Km1 + 1 - k))  # - np.log(Km1 - k)
	z = invlogit(y + eq_share)
	yl = np.concatenate([z, np.ones(y[:1].shape)])
	yu = np.concatenate([np.ones(y[:1].shape), 1 - z])
	S = cumprod(yu)
	x = S * yl
	return x.T

def stick_jacobian_det(y_):
	y = y_.T
	Km1 = y.shape[0]
	k = np.arange(Km1)[(slice(None), ) + (None, ) * (y.ndim - 1)]
	eq_share = logit(1. / (Km1 + 1 - k))  # -np.log(Km1 - k)
	yl = y + eq_share
	yu = np.concatenate([np.ones(y[:1].shape), 1 - invlogit(yl)])
	S = cumprod(yu)
	return np.sum(np.log(S[:-1]) - np.log1p(np.exp(yl)) - np.log1p(np.exp(-yl)), 0).T
