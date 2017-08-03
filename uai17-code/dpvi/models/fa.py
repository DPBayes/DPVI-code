from __future__ import division

import autograd.numpy as np
from autograd.core import primitive
from autograd.scipy.special import gammaln
from autograd.scipy.stats import multivariate_normal, dirichlet
from autograd.scipy.misc import logsumexp
from utils.misc import softmax, softplus
from utils.advi import draw_samples
from scipy.stats import gamma


@primitive
def lgamma(x,alpha,beta):
	return gamma.logpdf(x,alpha,scale=beta)
lgamma.defvjp(lambda g, ans, vs, gvs, x, a, scale: g * ((a-1)/x - 1/scale), argnum=0)

def multi_lpdf(value, mean, cov):
	d = mean.shape[0]
	tmp2 = value-mean
	if value.ndim>1:
		return -0.5*(d*np.log(2*np.pi) + np.linalg.slogdet(cov)[1] + np.diag(np.dot(tmp2, np.dot(np.linalg.inv(cov),tmp2.T))))
	else:
		return -0.5*(d*np.log(2*np.pi) + np.linalg.slogdet(cov)[1] + np.dot(tmp2, np.dot(np.linalg.inv(cov),tmp2.T)))

#def multi_lpdf2(value, mean, W, tau):
	#d = mean.shape[0]
	#q = W.shape[1]
	#tmp = value-mean
	#d_minus = np.eye(d)*tau 
	##invcov = d_minus - (d_minus.dot(W.dot(np.linalg.inv(np.eye(q)+(W.T).dot(d_minus.dot(W)))))).dot(W.T.dot(d_minus))
	#invcov = d_minus - np.dot(d_minus,np.dot(W,np.dot(np.linalg.inv(np.eye(q)+tau*np.dot(W.T,W)),W.T)))*tau
	##logdetcov = -d*np.log(tau)-np.linalg.slogdet(np.dot(np.eye(q)-np.dot(W.T, np.linalg.inv(W.dot(W.T)+np.eye(d)/tau)), W))
	#logdetcov = np.linalg.slogdet(np.dot(W,W.T)+np.eye(d)/tau)
	#if value.ndim>1:
		##return -0.5*(d*np.log(2*np.pi) + np.linalg.slogdet(cov)[1] + np.diag(np.dot(tmp, np.dot(np.linalg.inv(cov),tmp.T))))
		#return -0.5*(d*np.log(2*np.pi) + logdetcov + np.diag(np.dot(tmp, np.dot(invcov,tmp.T))))
	#else:
		##return -0.5*(d*np.log(2*np.pi) + np.linalg.slogdet(cov)[1] + np.dot(tmp, np.dot(np.linalg.inv(cov),tmp.T)))
		#return -0.5*(d*np.log(2*np.pi) + logdetcov + np.dot(tmp, np.dot(logdetcov,tmp.T)))

def log_like(var_par, draw, data, q):
	if data.ndim>1:
		d = data.shape[1]
	else: d = data.shape[0]
	l = int(len(var_par)/2)
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu
	mus = samples[:d]
	alphas = np.exp(samples[d:d+q])
	tau = np.exp(samples[d+q])
	#tau = np.exp(samples[d])
	W = samples[d+q+1:].reshape([d,q])
	tmp = np.dot(W,W.T)+np.eye(d)/tau
	#logp = multi_lpdf2(data, mus, W, tau)
	logp = multi_lpdf(data, mus, tmp)
	return np.sum(logp)

def logprior(var_par, draw, q, a=None, b=None, d=None):
	# Log of prior probabilities
	l = int(len(var_par)/2)
	#d = int((l-1-q)/(1+q))
	#d = int((l-1)/(1+q))
	mu, cov = var_par[:l], np.exp(var_par[l:])
	samples = draw*cov + mu

	mus = samples[:d]
	alphas = samples[d:d+q]
	tau = samples[d+q]
	#tau = np.exp(samples[d])
	W = samples[d+q+1:].reshape([d,q])
	#W = samples[d+1:].reshape([q,d])
	return multi_lpdf(mus, mean = np.zeros(d), cov=1e3*np.eye(d)) + sum([multi_lpdf(W[:,i], mean = np.zeros(d), cov=np.eye(d)/alpha) 
														for i, alpha in enumerate(np.exp(alphas))]) +sum(lgamma(np.exp(alphas), 1e-3, 1e3) + alphas) + lgamma(np.exp(tau), 1e-3, 1e3) + tau
	#return multi_lpdf(mus, mean = np.zeros(d), cov=1e3*np.eye(d)) + sum([multi_lpdf(W[i], mean = np.zeros(d), cov=np.eye(d)) for i in range(q) 
														#]) + lgamma(np.exp(tau), 1e-3, 1e3) + tau

def pred_like(data, var_par, samples, q):
	plike = 0
	for y in data:
		logpp = []
		for s in samples:
			logpp.append(log_like(var_par, s, y, q))
		plike += logsumexp(logpp)-np.log(samples.shape[0])
	return plike/data.shape[0]


def EM(data_cov, n_samples, n_iter, q):
	n = n_samples
	Cyy = n*data_cov
	d = Cyy.shape[1]
	gamma_k = np.ones([d,q])
	omega_k = np.eye(q)
	W_k = np.ones([d,q])
	D_k = np.eye(d)
	for i in range(n_iter):
		
		D_k_minus = np.linalg.inv(D_k)
		bbT_d_inv = D_k_minus-D_k_minus.dot(W_k.dot(np.linalg.inv(np.eye(q)+(W_k.T).dot(D_k_minus.dot(W_k))))).dot(np.dot(W_k.T,D_k_minus))
		
		#E-step
		gamma_k = bbT_d_inv.dot(W_k)
		omega_k = np.eye(q)-(gamma_k.T).dot(W_k)
		
		# M-step
		W_k = Cyy.dot(gamma_k.dot(np.linalg.inv(np.dot(gamma_k.T,Cyy.dot(gamma_k))+n*omega_k)))
		D_k = (1/n)*np.eye(d)*(Cyy-np.dot(Cyy, gamma_k.dot(W_k.T)))

	return W_k, D_k, gamma_k, omega_k