from __future__ import division
from autograd import grad, jacobian
import autograd.numpy as np
import autograd.numpy.random as npr
from pathos.pools import ProcessPool
from itertools import repeat

nodes = 6
eps = 1e-1
n_mc = 1

def clip(x, c, axis):
	return (x.T/(np.linalg.norm(x, axis=axis)+1e-15)*np.clip(np.linalg.norm(x, axis=axis), 0, c)).T

def entro(var_par):
	l = int(var_par.shape[0]/2)
	return 0.5*l*(1+np.log(2.0*np.pi)) + 0.5*np.sum(var_par[l:])

def dp_advi(data, logl, logprior, T, start, C, B, eta, sigma, k):
	N = data.shape[0]
	l = int(start.shape[0]/2)
	var_par = np.copy(start)
	likeg = grad(logl)
	priorgrad = grad(logprior)
	entrograd = grad(entro)
	ada_g = np.zeros(2*l)+eps

	def clipper(par, draw, add, value):
		return clip((N/B)*likeg(par, draw, value, k)+add, C, 0)

	for i in range(T):
		if i % 10 == 0:
			print(i)
			ada_g = np.zeros(2*l)+eps
		likeg_tot = np.zeros(2*l)
		pg = np.zeros(2*l)
		indx = np.random.choice(N, B)
		entro_ = entrograd(var_par)
		for m in range(n_mc):
			draw = npr.randn(l)
			pg = priorgrad(var_par, draw, k)
			if(sigma != 0): 
				res = ProcessPool(nodes).map(clipper, repeat(var_par), repeat(draw), repeat(pg/B+entro_/(B*n_mc)), data[indx])
				likeg_tot += np.sum(np.array(res), 0)
			else:
				likeg_tot += logl(var_par, draw, data[indx], k)
			#pg += priorgrad(var_par, draw, k)
		z = 2*C*sigma*npr.randn(2*l)
		if sigma == 0.0:
			g = ((N/B)*likeg_tot + pg)/n_mc + entrograd_
		else: 
			g = likeg_tot/n_mc+z
		ada_g += g**2
		if sum(np.isnan(eta*g/np.sqrt(ada_g+0.001*np.ones(2*l))))!=0:
			print('NaNs occured at iteration: %d' %i)
			break
		var_par += eta*g/np.sqrt(ada_g+0.001*np.ones(2*l))
	return var_par

def draw_samples(var_par, n_samples):
	l = int(var_par.shape[0]/2)
	return var_par[:l] + np.exp(var_par[l:])*npr.randn(n_samples,l)