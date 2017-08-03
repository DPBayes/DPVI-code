from __future__ import division
import autograd.numpy as np
from utils.advi import dp_advi
from models.gaus_mix import logl, logprior, get_pars, pred_like
from utils.misc import stick_backward, ma, find_sigma_act
import matplotlib.pyplot as plt
import pickle

############################ JMLR data

rng = np.random.RandomState(123)

n_samples = 2000

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



T = 1000
C = 1.0
q = 0.003
sigma = 2.74
k = 5
l = 2*k-1+k*d
eta = 0.2*np.ones(2*l)
eta[l-k:l] = 0.05
eta[-k:] = 0.05
eta[:k-1] = 0.01
eta[l:l+k-1] = 0.01
start = 0.0*np.ones(2*l)
B = int(q*n_samples)


############################ Circle mixture plot

var_par = dp_advi(data, logl, logprior, T, start, C, B, eta, sigma, k)

# Save the parameters of variational posterior
pickle.dump(var_par, open('./data/var_par.p', 'wb'))
var_par = pickle.load(open('./data/var_par.p', 'rb'))

pars = get_pars(var_par[:l], k, d)
pi = stick_backward(pars['pi'])
mus = pars['mu'].reshape([k,d])
taus = np.exp(pars['tau'])


colors = ['red', 'blue', 'yellow', 'orange', 'turquoise']

from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('mog_advi.pdf')

fracs = np.argsort(pi)[-5:]
mus = mus[fracs,:]
taus = taus[fracs]

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
	circle.append(plt.Circle(mus[n], v[0], color = color, fill=False, linestyle = 'dashed'))
	true_circle.append(plt.Circle(ms[n], v_true[0], color = 'black', fill=False))

fig, ax = plt.subplots(figsize=(75.0/25.4, 75/25.4)) 
ax = plt.gca()
ax.cla() # clear things for fresh plot
ax.axis('equal')

ax.set_xlim((-6, 6))
ax.set_ylim((-6, 6))


ax.scatter(data[:, 0], data[:, 1], color = 'green', alpha = 0.2, s=2.0)

for n, color in enumerate(colors):
	ax.add_artist(circle[n])
	ax.add_artist(true_circle[n])
	ax.plot(mus[n][0], mus[n][1], 'o', color=color)
	ax.plot(ms[n][0], ms[n][1], 'o', color='black')


eps_ma = 2*ma(2*sigma, 0.5*1e-3, T, q)

plt.rcParams.update({'font.size': 6.0})
plt.title("GMM with synthetic data \n DPVI: " + r'$\epsilon_{ACT}=1.0$'+ r', $\epsilon_{MA} = %.3f$' %eps_ma +r', $\delta_{tot} = $' + format(1e-3)  )
plt.tight_layout()


pp.savefig(fig)
pp.close()
plt.close()

############################ Data for prediction likelihood plot

epsilons = np.geomspace(0.1, 10, 5)
iter_sigmas = np.empty(len(epsilons))
for i, epsilon in enumerate(epsilons):
	iter_sigmas[i] = find_sigma_act(epsilon, 1e-3, T, q)[0]

iter_sigmas = np.array([0.27, 0.34, 0.48, 1.05, 2.0])



n_ave = 5
pred_likes = np.zeros([len(iter_sigmas), n_ave])
for i, sigma in enumerate(iter_sigmas):
	print sigma
	for n in range(n_ave):
		print n
		par = dp_advi(data, logl, logprior, T, start, C, B, eta, sigma, k)
		pred_likes[i, n] = pred_like(par, test_data, k)


np_pred_likes = []
np_T = 2000
np_eta = 0.01
for n in range(n_ave):
	print n
	par = dp_advi(data, logl, logprior, np_T, start, C, B, np_eta, 0, k)
	np_pred_likes.append(pred_like(par, test_data, k))


pickle.dump([pred_likes/n_test, iter_sigmas], open('../dp_sgld/data/mog_preds_advi.p','wb'))
pickle.dump(np.array(np_pred_likes)/n_test, open('../dp_sgld/data/ndp_mog_preds.p','wb'))

