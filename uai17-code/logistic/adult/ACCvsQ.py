# Plot script for Accuracy vs sample size

from __future__ import division
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import sys
sys.path.append('/home/jalko/Dropbox/code/icml/inferences/logistic')
from bin_class import classifier as cl

T = 2000
T_nondp = 20000
delta = 1e-3
C = 50

data = pickle.load(open('./data/adult.p','rb'))

num_eps = 5

iter_sigma = np.linspace(0.4, 2, num_eps)
iter_q = np.linspace(0.001, 0.005, 5)
tot_eps = np.zeros([num_eps,5])
sd = np.zeros([num_eps,5])
acc = np.zeros([num_eps,5])

n_ave = 5

for i in range(0,num_eps):
	print i
	for j in range(0,5):
		sigma = iter_sigma[i]
		q = iter_q[j]
		tmp_acc = []
		for k in range(0,n_ave):
			acc_run, eps = cl.advi(data=data, T=T, q=q, learning_rate=0.1, C=C, delta=delta, sigma=sigma)
			tmp_acc.append(acc_run)
		acc[i,j] = np.mean(tmp_acc)
		sd[i,j] = np.sqrt(np.var(tmp_acc)/n_ave)
		tot_eps[i,j] = eps

#Dump plotting data
pickle.dump([iter_q,acc,sd,tot_eps],open('./data/adult_acc_vs_q.p','wb'))

#Non private reference
q = 0.001
np_acc = cl.advi(data=data, T=T_nondp, q=q, learning_rate=0.01)
pickle.dump([np_acc], open('./data/ndp.p','wb'))


[np_acc] = pickle.load(open('./data/ndp.p','rb'))
[iter_c,acc,sd,tot_eps] = pickle.load(open('./data/adult_acc_vs_q.p', 'rb'))

pp = PdfPages('./plots/adult_acc_vs_q.pdf')
plot1 = plt.figure(figsize=(75.0/25.4, 75/25.4))
plt.rcParams.update({'font.size': 6.0})
for j in range(0,5):
	plt.errorbar(tot_eps[:,j], acc[:,j], yerr = sd[:,j],label = "$q$="+str(iter_q[j]))
plt.axhline(np_acc, color = "red", linestyle="dashed",label="Non DP")
plt.xlabel(r'$\epsilon_{tot}$')
plt.ylabel('Test accuracy')
plt.legend(loc="lower right")
plt.title("Adult data \n" +"T={}, C={}, ".format(T,C)+r'$\delta_{tot}=$'+format(delta))
plt.tight_layout()
plt.xscale('log')
pp.savefig(plot1)
plt.savefig("./plots/adult_acc_vs_q.png")
pp.close()
plt.close()