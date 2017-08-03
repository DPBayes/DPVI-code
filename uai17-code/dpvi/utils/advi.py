from __future__ import division
from autograd import grad, jacobian
import autograd.numpy as np
import autograd.numpy.random as npr
from pathos.pools import ProcessPool
from itertools import repeat

nodes = 6
eps = 1e-1
n_mc = 4
clear_advi = 10

rng = npr.RandomState(123)

def clip(x, c, axis):
	return (x.T/(np.linalg.norm(x, axis=axis)+1e-15)*np.clip(np.linalg.norm(x, axis=axis), 0, c)).T

def entro(var_par):
	l = int(var_par.shape[0]/2)
	return 0.5*l*(1+np.log(2.0*np.pi)) + 0.5*np.sum(var_par[l:])

def dp_advi(data, logl, logprior, T, start, C, B, eta, sigma, *args, **kwargs):
	N = data.shape[0]
	l = int(start.shape[0]/2)
	var_par = np.copy(start)

	likeg = grad(logl)
	priorgrad = grad(logprior)
	entrograd = grad(entro)
	def clipper(par, draw, value):
		#return clip((N/B)*likeg(par, draw, value, *args), C, 0)
		return clip(likeg(par, draw, value, *args), C, 0)

	for i in range(T):
		if i % clear_advi == 0:
			print(i)
			ada_g = np.zeros(2*l)+eps
		likeg_tot = np.zeros(2*l)
		pg = np.zeros(2*l)
		indx = rng.choice(N, B)
		entrograd_ = entrograd(var_par)

		for m in range(n_mc):
			draw = rng.randn(l)
			if(sigma != 0): 
				res = ProcessPool(nodes).map(clipper, repeat(var_par), repeat(draw), data[indx])
				#likeg_tot += np.sum(np.array(res), 0)
				likeg_tot += (N/B)*np.sum(np.array(res), 0)
			else:
				likeg_tot += (N/B)*likeg(var_par, draw, data[indx], *args)
			pg += priorgrad(var_par, draw, *args, **kwargs)
		
		z = (N/B)*2*C*sigma*rng.randn(2*l)
		g = (likeg_tot + pg)/n_mc + entrograd_+z
		
		ada_g += g**2
		if sum(np.isnan(eta*g/np.sqrt(ada_g+0.001*np.ones(2*l))))!=0:
			print('NaNs occured at iteration: %d' %i)
			break
		var_par += eta*g/np.sqrt(ada_g+0.001*np.ones(2*l))
	return var_par

def draw_samples(var_par, n_samples):
	l = int(var_par.shape[0]/2)
	return var_par[:l] + np.exp(var_par[l:])*npr.randn(n_samples,l)