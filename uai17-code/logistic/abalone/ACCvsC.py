# Plot script for Accuracy vs. Clipping threshold 


from __future__ import division
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
sys.path.append('../')
from bin_class import classifier as cl

T = 2000
T_nondp = 20000
delta = 1e-3
q = 0.01

data = pickle.load(open('./data/aba.p','rb'))

num_eps = 5
iter_sigma = np.linspace(2, 10, num_eps)
iter_c = np.linspace(1, 5, 5)
tot_eps = []
sd = np.zeros([num_eps,5])
acc = np.zeros([num_eps,5])

# Number of runs to take average on
n_ave = 5

for i in range(0,num_eps):
	print i
	for j in range(0,5):
		sigma = iter_sigma[i]
		C = iter_c[j]
		tmp_acc = []
		for k in range(0,n_ave):
			acc_run, eps = cl.advi(data=data, T=T, q=q, learning_rate=0.1, C=C, delta=delta, sigma=sigma)
			tmp_acc.append(acc_run)
		acc[i,j] = np.mean(tmp_acc)
		sd[i,j] = np.sqrt(np.var(tmp_acc)/n_ave)
	tot_eps.append(eps)

#Dump plotting data
pickle.dump([iter_c,acc,sd,tot_eps],open('./data/abalone_acc_vs_c.p','wb'))

# Non private reference
np_acc = cl.advi(data=data, T=T_nondp, q=q, learning_rate=0.01)
pickle.dump([np_acc], open('./data/abalone_ndp.p','wb'))

[np_acc] = pickle.load(open('./data/abalone_ndp.p','rb'))
[iter_c,acc,sd,tot_eps] = pickle.load(open('./data/abalone_acc_vs_c.p', 'rb'))

pp = PdfPages('./plots/abalone_acc_vs_c.pdf')
plot1 = plt.figure(figsize=(75.0/25.4, 75/25.4))
plt.rcParams.update({'font.size': 6.0})

for j in range(0,5):
	plt.errorbar(tot_eps, acc[:,j], yerr = sd[:,j],label = "$c_{t}$="+str(iter_c[j]))
plt.axhline(np_acc, color = "red", linestyle="dashed",label="Non DP")
plt.xlabel(r'$\epsilon_{tot}$')
plt.ylabel('Test accuracy')
plt.legend(loc="lower right")
plt.title("Abalone data \n" +"T={}, q={} ,".format(T,q)+r'$\delta_{tot}=$'+format(delta))
plt.ylim(0.7, 0.775)
plt.tight_layout()
plt.xscale('log')
pp.savefig(plot1)
plt.savefig("./plots/abalone_acc_vs_c.png")
pp.close()
plt.close()