from active_work.miscellaneous import RTPring
from active_work.plot import list_colormap
from active_work.init import get_env

import matplotlib.pyplot as plt
plt.style.use('paper')
from matplotlib.lines import Line2D

import numpy as np

r = RTPring()

_L = [.5, 2, 10]

x = {L: np.array(sorted(np.linspace(0, np.max([-2, -1./L]), 1000, endpoint=False)) + np.linspace(0, 5, 1000).tolist()) for L in _L}
s = {L: np.array(list(map(lambda _: r.s(L, _), x[L])))*L for L in _L}
nu = {L: np.array(list(map(lambda _: r.nu(_), x[L]))) for L in _L}
nuAve = {L: np.array(list(map(lambda _: r.nuAve(L, _), x[L]))) for L in _L}

smin = -10
xmax = 0.5

# plot

colors = list_colormap(_L, sort=True)
adjust = {'left': 0.32, 'right': 0.99, 'bottom': 0.20, 'top': 0.95}

# psi

fig, ax = plt.subplots()
ax.set_xlabel(r'$\lambda L v_0$')
ax.set_xlim([-10, 10])
ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
ax.set_xticklabels(['', r'$-8$', '', r'$-4$', '', r'$0$', '', r'$4$', '', r'$8$', ''])
ax.set_ylabel(r'$\tau_{\rm p} \psi^{\rm RTP}(\lambda)$')

line = {}
for L in _L:
	line[L], = ax.plot(s[L][(x[L] < xmax)*(s[L] > smin)], x[L][(x[L] < xmax)*(s[L] > smin)],
		color=colors[L], label=r'$\tilde{L} = %s$' % L)
ax.axvline(x=0, color='black', linewidth=2)
ax.axhline(y=0, color='black', linewidth=2)

plt.sca(ax)
ax.add_artist(plt.legend(loc='lower right', ncol=1, borderpad=0.025, handletextpad=0.2,
	handles=[
		Line2D([0], [0], lw=0, label=r'$L/l=$'),
		*[Line2D([0], [0], color=line[L].get_color(), label=r'$%s$' % L)
			for L in _L]]))
fig.subplots_adjust(**adjust)

# nu

fign, axn = plt.subplots()
axn.set_xlabel(r'$\lambda L v_0$')
axn.set_xlim([-10, 10])
axn.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
axn.set_xticklabels(['', r'$-8$', '', r'$-4$', '', r'$0$', '', r'$4$', '', r'$8$', ''])
axn.set_ylabel(r'$\nu^{\rm RTP}(\lambda)$')
axn.set_ylim([0, 1])

linen = {}
for L in _L:
#	linen[L], = axn.plot(s[L][(x[L] < xmax)*(s[L] > smin)], nu[L][(x[L] < xmax)*(s[L] > smin)],
	linen[L], = axn.plot(s[L][(s[L] > smin)], nu[L][(s[L] > smin)],
		color=colors[L], linestyle='--', label=r'$\tilde{L} = %s$' % L) 
	linen[L], = axn.plot(s[L][(s[L] > smin)], nuAve[L][(s[L] > smin)],
		color=colors[L], linestyle='-', label=r'$\tilde{L} = %s$' % L)
axn.axvline(x=0, color='black', linewidth=2)
axn.axhline(y=0.5, color='black', linewidth=2)

plt.sca(axn)
axn.add_artist(plt.legend(loc='upper right', ncol=1, borderpad=0.025, handletextpad=0.2, labelspacing=0.2,
	handles=[
		Line2D([0], [0], lw=0, label=r'$L/l=$'),
		*[Line2D([0], [0], color=linen[L].get_color(), label=r'$%s$' % L)
			for L in _L]]))
axn.add_artist(plt.legend(loc='lower left', ncol=1, borderpad=0.025, handletextpad=0.2,
	handles=[
		Line2D([0], [0], color='black', linestyle='--',
			label=r'$\nu^{\rm RTP}_{\rm end}$'),
		Line2D([0], [0], color='black', linestyle='-',
			label=r'$\nu^{\rm RTP}_{\rm ave}$')]))
fign.subplots_adjust(**adjust)

# show and save

if get_env('SAVE', default=False, vartype=bool):
	def save(f, fname):
		f.savefig(fname + '.eps')
		f.savefig(fname + '.svg')
	save(fig, 'exactPsiRTP')
	save(fign, 'exactNuRTP')

if get_env('SHOW', default=True, vartype=bool): plt.show()

