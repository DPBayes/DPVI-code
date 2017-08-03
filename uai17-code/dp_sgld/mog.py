from __future__ import division
import autograd.numpy as np
from dp_sgld import dp_sgld
from gmix import logl, logprior, softmax, pred_like
import matplotlib.pyplot as plt
import pickle


##################### JMLR synthetic data

n_samples = 2000
rng = np.random.RandomState(123)
R = 2
d = 2
ms = np.array([[0, 0], [R, R],[-R, -R],[-R, R],[R, -R]])
k = len(ms)
ps = np.ones(k)/k
ts = 0.5*np.ones(k)

zs = np.array([rng.multinomial(1, ps) for _ in range(n_samples)]).T
xs = [z[:, np.newaxis] * rng.multivariate_normal(m, t*np.eye(2), size=n_samples)
      for z, m, t in zip(zs, ms, ts)]
data = np.sum(np.dstack(xs), axis=2)


n_test = 100
test_zs = np.array([rng.multinomial(1, ps) for _ in range(n_test)]).T
test_xs = [z[:, np.newaxis] * rng.multivariate_normal(m, t*np.eye(2), size=n_test)
      for z, m, t in zip(test_zs, ms, ts)]
test_data = np.sum(np.dstack(test_xs), axis=2)



T = 4.5
L = 1
q = 0.03
model = 'mvn_mix' + str(k)
B = int(q*n_samples)
start = 0.0*np.ones(k*d + 2*k)
eta = 1e-4
delta = 1e-3

##################### Circle plot

samples = dp_sgld(data, logl, logprior, T, start, L, int(q*n_samples), eta, [1.0, delta], model)
pickle.dump([samples, pred_like(samples[-100:], test_data, k)], open('./data/mog_sgld_samples.p','wb'))
samples, pxpl = pickle.load(open('./data/mog_sgld_samples.p','rb'))
samples_a = samples[-100:]
mus = samples_a[:,k:-k]
mu_mean = np.mean(mus, axis=0).reshape([k,d])
log_taus = samples_a[:,-k:]
taus = np.exp(log_taus)
tau_mean = np.mean(taus, axis=0)

colors = ['red', 'blue', 'yellow', 'orange', 'turquoise']

from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('./plots/mog_sgld.pdf')


circle = []
true_circle = []
for n, color in enumerate(colors):
	v, w = np.linalg.eigh(taus[n]*np.eye(k))
	v_true, w_true = np.linalg.eigh(ts[n]*np.eye(k))
	u = w[0] / np.linalg.norm(w[0])
	u_true = w_true[0] / np.linalg.norm(w_true[0])
	angle = np.arctan2(u[1], u[0])
	angle_true = np.arctan2(u_true[1], u_true[0])
	angle = 180 * angle / np.pi
	angle_true = 180 * angle_true / np.pi
	v = 2. * np.sqrt(2.) * np.sqrt(v)
	v_true = 2. * np.sqrt(2.) * np.sqrt(v_true)
	circle.append(plt.Circle(mu_mean[n], v[0], color = color, fill=False, linestyle = 'dashed'))
	true_circle.append(plt.Circle(ms[n], v_true[0], color = 'black', fill=False))

fig, ax = plt.subplots(figsize=(75.0/25.4, 75/25.4)) 
plt.rcParams.update({'font.size': 6.0})
ax = plt.gca()
ax.cla() # clear things for fresh plot
ax.axis('equal')

ax.set_xlim((-6, 6))
ax.set_ylim((-6, 6))


ax.scatter(data[:, 0], data[:, 1], color = 'green', alpha = 0.2, s=2.0)
for n, color in enumerate(colors):
	ax.add_artist(circle[n])
	ax.add_artist(true_circle[n])
	ax.plot(mu_mean[n][0], mu_mean[n][1], 'o', color=color)
	ax.plot(ms[n][0], ms[n][1], 'o', color='black')



plt.title("GMM with synthetic data \n SGLD: "+r' $\epsilon=1.0$'+r', $\delta_{tot} = $' + format(delta)  )
plt.tight_layout()

pp.savefig(fig)
pp.close()
plt.close()


#################### Plot prediction likelihoods 


import numpy as np
px_pred_likes_advi, iter_sigmas = pickle.load(open('./data/mog_preds_advi.p','rb'))
from misc import act, ma
T_advi = 1000
q_advi = 0.003
delta = 1e-3
eps_tot = np.geomspace(0.1, 10, 5)
eps_tot_ma = []
for sigma in iter_sigmas:
	eps_tot_ma.append(2*ma(2*sigma, 0.5*delta, T_advi, q_advi, max_moment=200))

n_ave = 5
pred_likes = np.zeros([len(eps_tot), n_ave])
for i, eps in enumerate(eps_tot):
	print eps
	for n in range(n_ave):
		print n
		samples = dp_sgld(data, logl, logprior, T, start, L, B, eta, [eps, 1e-3], model)
		pred_likes[i, n] = pred_like(samples[-100:], test_data, k)


pickle.dump([pred_likes, eps_tot], open('./data/mog_preds_sgld.p','wb'))
pred_likes, eps_tot = pickle.load(open('./data/mog_preds_sgld.p','rb'))
np_pred = pickle.load(open('./data/ndp_mog_preds.p','rb'))
px_pred_likes_sgld = pred_likes/n_test

px_pred_means_sgld = np.mean(px_pred_likes_sgld, axis=1)
px_pred_means_advi = np.mean(px_pred_likes_advi, axis=1)

px_pred_sds_sgld = np.sqrt(np.var(px_pred_likes_sgld, axis=1)/px_pred_likes_sgld.shape[1])
px_pred_sds_advi = np.sqrt(np.var(px_pred_likes_advi, axis=1)/px_pred_likes_advi.shape[1])

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('./plots/predlike_mog_sgld_vs_advi.pdf')

plot1 = plt.figure(figsize=(75.0/25.4, 75/25.4))
plt.rcParams.update({'font.size': 6.0})

plt.errorbar(eps_tot,px_pred_means_sgld, yerr=px_pred_sds_sgld, label='DP-SGLD')
plt.errorbar(eps_tot_ma,px_pred_means_advi, yerr=px_pred_sds_advi, label='DPVI-MA')
plt.errorbar(eps_tot,px_pred_means_advi, yerr=px_pred_sds_advi, label='DPVI')
plt.axhline(np.mean(np_pred), color = "red", linestyle="dashed",label="Non DP")

plt.xlabel(r'$\epsilon_{tot}$')
plt.ylabel('Per Example Predictive likelihood')
plt.legend(loc="lower right")
plt.title("GMM with synthetic data")
plt.tight_layout()
plt.xscale('log')
pp.savefig(plot1)
plt.savefig("./plots/predlike_mog_sgld_vs_advi.png")
pp.close()
plt.close()
