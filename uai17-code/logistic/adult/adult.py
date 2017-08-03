# Adult example

from __future__ import division
import numpy as np
import pickle

import sys
sys.path.append('../')
from bin_class import classifier as cl

sys.path.append('/home/jalko/Dropbox/code/icml/inferences/logistic')
from bin_class import classifier as cl


np.random.seed(123)

def sigmoid(x):
	lower = 1e-6
	upper = 1 - 1e-6
	return lower + (upper - lower) * np.exp(x) / (1 + np.exp(x))


data = pickle.load(open('./data/adult.p','rb'))

T = 2000 #  Number of ADVI iterations
sigma = 10 # Privacy parameters
C = 75
delta = 1e-3
q = 0.005 

n_ave = 10
iter_sigmas = np.exp(-np.linspace(-3,0.3,10))
accs = np.empty([iter_sigmas.shape[0], n_ave])

for k in range(iter_sigmas.shape[0]):
	sigma = iter_sigmas[k]

	for i in range(n_ave):
		accs[k,i] = cl.advi(data=data, T=T, q=q, learning_rate=0.1, C=C, delta=delta, sigma=sigma)[0]

pickle.dump([accs,iter_sigmas], open('../../dp_sgld/data/adult_advi_acc_sigmas.p','wb'))