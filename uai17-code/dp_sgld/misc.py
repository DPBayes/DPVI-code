from __future__ import division
import numpy as np
#import autograd.numpy as np

import sys
sys.path.append('/home/jalko/Dropbox/code/icml')
from ma import gaussian_moments as gm

def act(sigma, delta_tot, delta_prime, T, q):
	d_iter = (delta_tot-delta_prime)/(T*q)
	sigma_prime = np.log(1+q*(np.exp(np.sqrt(2*np.log(1.25/d_iter))/sigma)-1))
	return np.sqrt(2*T*np.log(1.25/delta_prime))*sigma_prime+T*sigma_prime*(np.exp(sigma_prime)-1)

def ma(sigma, delta, T, q, max_moment=100):
	lm = []
	for i in range(1,max_moment):
		tmp = gm.compute_log_moment(q, sigma, T, i)
		lm += [(i, tmp)]
	return gm._compute_eps(lm, delta) 
