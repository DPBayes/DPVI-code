from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.stats import multivariate_normal as mvn

npr.seed(123)

def sigmoid(x):
	lower = 1e-6
	upper = 1 - 1e-6
	return lower + (upper - lower) * np.exp(x) / (1 + np.exp(x))

def logl(sample, data):
	#Log likelihood for logistic regression
	x,y = data[:, :-1], data[:, -1]
	return -np.sum(1*np.log(1+ np.exp(-y*np.dot(x,sample))), axis = 0)

def logprior(sample):
	# Assume 0-centered MV-normal prior with covariance matrix I
	return np.sum(mvn.logpdf(sample, mean=np.zeros(sample.shape), cov=np.diag(np.ones(sample.shape))))

def predict(test_data, samples):
	x_test, y_test = test_data
	pred = 2*(np.mean(sigmoid(np.dot(x_test, samples.T)), axis=1)>0.5)-1
	return np.mean(y_test==pred)
