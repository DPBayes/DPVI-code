from __future__ import division
import autograd.numpy as np


import pickle
data = pickle.load(open('../logistic/abalone/data/aba.p','rb'))

y, x = data
N, D = x.shape

N_train = int(0.8*N)
N_test = N-N_train

x_train = x[:N_train,:]
y_train = 2*y[:N_train]-1
x_test = x[N_train:,:]
y_test = 2*y[N_train:]-1

train_data = np.empty([N_train, D+1])
train_data[:,:-1] = x_train
train_data[:,-1] = y_train

# Values from DPVI experiment
[accs_advi, iter_sigmas] = pickle.load(open('./data/aba_advi_acc_sigmas.p', 'rb'))
[np_acc] = pickle.load(open('../logistic/abalone/data/abalone_ndp.p','rb'))


from dp_sgld import dp_sgld

T_advi = 1000
q = 0.05
B = int(np.floor(q*N_train))
T = 10
L = 5 
start = np.zeros(D)
model = 'logistic'
from misc import act, ma
delta = 1e-3
delta_prime = 4*1e-5 
eps_tot = 2*act(2*iter_sigmas, 0.5*delta, delta_prime, T_advi, q)
n_ave = 10
i = 0
accs_sgld = np.empty([eps_tot.shape[0],n_ave])
from logistic import logl, logprior, predict

for eps in eps_tot:
	for m in range(0,n_ave):
		print m
		start = np.zeros(D)
		samples = dp_sgld(train_data, logl, logprior, T, start, L, B, 0.1, [eps, delta], model)
		samples_a = samples[-100:]
		accs_sgld[i,m] = predict([x_test, y_test], samples_a)
	i += 1

pickle.dump([accs_sgld, eps_tot, 0.001], open('./data/aba_sgld_acc.p', 'wb'))
[accs_sgld, eps_tot, delta] =  pickle.load(open('./data/aba_sgld_acc.p', 'rb'))
acc_advi = np.mean(accs_advi, axis=1)
sd_advi = np.sqrt(np.var(accs_advi, axis=1)/accs_advi.shape[1])

acc_sgld = np.mean(accs_sgld, axis=1)
sd_sgld = np.sqrt(np.var(accs_sgld, 1)/n_ave)

eps_tot_ma = []
for sigma in iter_sigmas:
	eps_tot_ma.append(ma(sigma, 0.5*delta, T_advi, q, 160))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('./plots/sgld_vs_advi_abalone.pdf')

plot1 = plt.figure(figsize=(75.0/25.4, 75/25.4))
plt.rcParams.update({'font.size': 6.0})

plt.errorbar(eps_tot,acc_sgld, yerr=sd_sgld, label='DP-SGLD')
plt.errorbar(eps_tot,acc_advi, yerr=sd_advi, label='DPVI')
plt.errorbar(eps_tot_ma,acc_advi, yerr=sd_advi, label='DPVI-MA')
plt.axhline(np_acc, color = "red", linestyle="dashed",label="Non DP")

plt.xlabel(r'$\epsilon_{tot}$')
plt.ylabel('Test accuracy')
plt.legend(loc="lower right")
plt.title("Abalone data")
plt.tight_layout()
plt.xscale('log')
pp.savefig(plot1)
plt.savefig("./plots/sgld_vs_advi_abalone.png")
pp.close()
plt.close()
