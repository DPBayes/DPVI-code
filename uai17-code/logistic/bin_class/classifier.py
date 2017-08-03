## Binary classifier based on logistic regression

from __future__ import division
import numpy as np
import pymc3 as pm
from pymc3.math import logsumexp
import theano.tensor as tt
import collections

import sys
sys.path.append('../../')
from dp_advi_pymc import advi_minibatch as ad
from ma import gaussian_moments as gm

np.random.seed(123)

def sigmoid(x):
	lower = 1e-6
	upper = 1 - 1e-6
	return lower + (upper - lower) * np.exp(x) / (1 + np.exp(x))

def advi(data, T, q, learning_rate=0.01, C=None, delta=None, sigma=None):
	"""
	Binary classifier
	input format:
	data, in [y,x] where y is binary vector and x matrix of features
	T, number of advi iterations
	q, sampling ratio
	learning_rate=None, C=None, delta=None, sigma=None Privacy parameters
	"""
	
	y, x = data
	N, D = x.shape
	
	N_train = int(0.8*N)
	N_test = N-N_train

	x_train = x[:N_train,:]
	y_train = y[:N_train]
	x_test = x[N_train:,:]
	y_test = y[N_train:]

	# Create PyMC3 model with subsampling
	B = int(q*N_train) # Batch size
	x_t = tt.matrix()
	y_t = tt.vector()
	x_t.tag.test_value = np.zeros((B, D)).astype(float)
	y_t.tag.test_value = np.zeros((B,)).astype(float)


	with pm.Model() as logistic_model:
		w = pm.MvNormal('w', mu = np.zeros(D), tau = np.eye(D), shape=(D,))
		y_obs = pm.Bernoulli('y_obs', p=sigmoid(tt.dot(x_t,w)), observed=y_t)


	def minibatch_gen(y, x):
		while True:
			ixs = np.random.choice(range(0,N_train),B)
			yield y[ixs], x[ixs]
			
	minibatches = minibatch_gen(y_train, x_train)

	# With privacy use advi_minibatch
	if(sigma!=None): means, sds, elbos = ad.advi_minibatch(vars=None, start=None, model=logistic_model, n=T, n_mcsamples=1,
					minibatch_RVs=[y_obs], minibatch_tensors=[y_t, x_t], minibatches=minibatches, total_size=N_train, learning_rate=learning_rate, verbose=0, dp_par = [sigma, C])

	# Non private version uses pm original advi
	else: means, sds, elbos = pm.advi_minibatch(vars=None, start=None, model=logistic_model, n=T, n_mcsamples=1,
					minibatch_RVs=[y_obs], minibatch_tensors=[y_t, x_t], minibatches=minibatches, total_size=N_train, learning_rate=learning_rate)

	# Laplace approx
	w = means['w']
	S = np.diag(sds['w'])**2

	mua=np.dot(x_test,w)
	sia=np.empty(N_test)
	for i in range(0, N_test):
		sia[i] = np.dot(np.dot(S,x_test[i]), x_test[i])
	preds = 1*(sigmoid(mua/np.sqrt(1+np.pi*sia/8))>=0.5)
	acc = np.sum(y_test==preds)/N_test


	# Privacy calculation
	if(sigma!=None):
		lm = []
		for i in range(1,100):
			tmp = gm.compute_log_moment(q, sigma, T, i)
			lm += [(i, tmp)]

		return acc, gm._compute_eps(lm, delta)
	else: return acc