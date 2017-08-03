from __future__ import division
from autograd import grad, jacobian
import autograd.numpy as np
import autograd.numpy.random as npr

rng = npr.RandomState(123)

def clip(x, c, axis=None):
	return (x.T/(np.linalg.norm(x, axis=axis)+1e-9)*np.clip(np.linalg.norm(x, axis=axis), 0, c)).T


def dp_sgld(data, logl, logprior,T, start, L, b, eta0, dp_budget, model):
	epsilon, delta = dp_budget
	N = data.shape[0]
	l = start.shape[0]
	sample = np.copy(start)
	samples = np.empty([int(T*N/b),l])
	priorgrad = grad(logprior)
	C = 128*N*T*L**2/(b*epsilon**2)*np.log(5*N*T/(2*b*delta))*np.log(2/delta) # LD constant

	if model == 'logistic':
		likegrad = grad(logl)
		for i in range(int(T*N/b)):
			eta = max(1/(i+1),eta0)
			indx = np.random.choice(N, b)
			z = np.sqrt(max(eta**2*C, eta))*rng.randn(l)
			sample += eta*((N/b)*likegrad(sample, data[indx]) + priorgrad(sample)) + z
			sample = clip(sample, L-1, 0)
			if sum(np.isnan(sample))==0:
				samples[i] = sample
			else: 
				print 'Nans'
				return samples[:i]
			#if i % 100 == 0: print(i)

	elif model == 'multi':
		likeJ = jacobian(logl)
		#likeg = grad(logl)
		for i in range(int(T*N/b)):
			eta = eta0
			indx = rng.choice(N, b)
			z = np.sqrt(max(eta**2*C, eta))*rng.randn(l)
			J = likeJ(sample, data[indx])
			tot_g = np.zeros(l)
			#for value in data[indx]:
				#tot_g += clip(likeg(sample, value), L)
			#sample = sample + eta*((N/b)*tot_g + priorgrad(sample)) - z
			sample += eta*((N/b)*np.sum(clip(J, L, 1), axis=0)  + priorgrad(sample) ) - z
			if sum(np.isnan(sample))==0:
				samples[i] = sample
			else: 
				print 'Nans'
				raise Error()
				return samples[:i]
			if i % 10 == 0: print(i)

	elif model[:-1] == 'mvn_mix':
		k = int(model[-1])
		#likeJ = jacobian(logl)
		likeg = grad(logl)
		for i in range(int(T*N/b)):
			eta = eta0
			indx = rng.choice(N, b)
			z = np.sqrt(max(eta**2*C, eta))*rng.randn(l)
			#J = likeJ(sample, k, data[indx])
			tot_g = np.zeros(l)
			for value in data[indx]:
				tot_g += clip(likeg(sample, k, value), L)
			sample = sample + eta*((N/b)*tot_g + priorgrad(sample, k)) - z
			#sample += eta*((N/b)*np.sum(clip(J, L, 1), axis=0)  + priorgrad(sample) ) - z
			if sum(np.isnan(sample))==0:
				samples[i] = sample
			else: 
				print 'Nans'
				return samples[:i]
			#if i % 10 == 0: print(i)

	return samples