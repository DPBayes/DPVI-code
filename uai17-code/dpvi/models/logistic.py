from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.stats import multivariate_normal as mvn
from utils.misc import softmax

npr.seed(123)

def logl(var_par, draw, data):
	l = int(var_par.shape[0]/2)
	sample = var_par[:l]+np.exp(var_par[l:])*npr.randn(l)
	#Log likelihood for logistic regression
	x,y = data[:, :-1], data[:, -1]
	return -1*np.log(1+ np.exp(-y*np.dot(x,sample)))

def logprior(var_par, draw):
	l = int(var_par.shape[0]/2)
	sample = var_par[:l]+np.exp(var_par[l:])*npr.randn(l)
	# Assume 0-centered MV-normal prior with covariance matrix I
	return np.sum(mvn.logpdf(sample, mean=np.zeros(sample.shape), cov=np.diag(np.ones(sample.shape))))

def predict(test_data, samples):
	x_test, y_test = test_data
	pred = 2*(np.mean(sigmoid(np.dot(x_test, samples.T)), axis=1)>0.5)-1
	return np.mean(y_test==pred)
