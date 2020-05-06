from active_work.rotors import Mathieu
from active_work.rotors import VariationalPolarisationSquared as VPS
from active_work.rotors import VariationalPolarisation as VP
from active_work.cloningR import CloningOutput as CO
from active_work.cloningR import filename
from active_work.plot import list_colormap
from active_work.init import get_env

import numpy as np

from sage.all import var, bessel_I, find_local_maximum, find_local_minimum, function

import matplotlib.pyplot as plt
plt.style.use('paper')
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.optimize import root_scalar
from scipy.misc import derivative

Dr = 0.5
mathieu = Mathieu(Dr)
vp = VP(Dr)
vps = VPS(Dr)

px = np.linspace(0, 0.5, 100)
theta = np.linspace(-np.pi/2, np.pi/2, 100)

# ANALYTICS

# rate function

figI, axI = plt.subplots()
axI.set_xlabel(r'$p_x$')
axI.set_ylabel(r'$I((p_x, 0))$')
axI.set_xlim([0, 0.5])
axI.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
axI.set_xticklabels([r'$0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$'])

I, = axI.plot(px, mathieu.rate(*[(p, 0) for p in px]),
	label=r'$I((p_x, 0))$')
Isq, = axI.plot(px, (1./2)*Dr*(px**2),
	label=r'$\frac{1}{2} \, D_r \, p_x^2$')

plt.sca(axI)
axI.add_artist(plt.legend(loc=2,
	handles=[
		Line2D([0], [0], color=I.get_color(),
			label=I.get_label()),
		Line2D([0], [0], color=Isq.get_color(),
			label=Isq.get_label())]))
axI.add_artist(plt.legend(loc=4,
	handles=[
		Line2D([0], [0], lw=0,
			label=r'$D_r=%.1f$' % Dr)]))
figI.subplots_adjust(left=0.24, bottom=0.20)

# potential

s = -1

figV, axV = plt.subplots()
axV.set_xlabel(r'$\theta$')
axV.set_ylabel(r'$\phi_s(\theta)$')
axV.set_xlim([-2, 2])
axV.set_xticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2])
axV.set_xticklabels([r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{3}$', r'$-\frac{\pi}{6}$', r'$0$', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$'])
axV.tick_params(axis='x', pad=10)

phi, = axV.plot(theta, mathieu.optimal_potential(s, *theta),
	label=r'$\phi_s$',
	zorder=2)
phig, = axV.plot(theta, mathieu.optimal_potential_curvature(s)*(1 - np.cos(theta)),
	label=r'$\phi^{(g)}_s$',
	zorder=1)
phisq, = axV.plot(theta, 0.5*(theta**2)*mathieu.optimal_potential_curvature(s),
	label=r'$\frac{1}{2} \, \theta^2 \, \partial^2_{\theta} \phi_s(\theta = 0)$',
	zorder=0)
phivp, = axV.plot(theta, 2*vp.h(s)[0]*(1 - np.cos(theta)),
#	label=r'$2 \, h_p(s) \, (1 - \cos\theta)$')
	label=r'$\phi_p(\theta)$')

plt.sca(axV)
axV.add_artist(plt.legend(loc=9,
	handles=[
		Line2D([0], [0], color=line.get_color(),
			label=line.get_label())
		for line in (phi, phig, phisq, phivp)]))
axV.add_artist(plt.legend(loc=4, fontsize=19,
	handles=[
		Line2D([0], [0], lw=0,
			label=r'$s=%.1f$' % s),
		Line2D([0], [0], lw=0,
			label=r'$D_r=%.1f$' % Dr)]))
figV.subplots_adjust(bottom=0.20, left=0.21)

# NUMERICS

_N = [10, 20, 50, 100]
nc = 1000
bias = 1
launch = 0

co = {N: CO(filename(N, Dr, nc, bias, launch) + '.cloR') for N in _N}
rate = {}
for N in co: _, _, _, rate[N] = co[N].meanSterr()

colors = list_colormap(_N, sort=True)

# rate function from cloning

figIN, axIN = plt.subplots()
axIN.set_xlabel(r'$p^2$')
axIN.set_ylabel(r'$I(p^2)$')
axIN.set_xlim([0, 0.5])
axIN.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
axIN.set_xticklabels([r'$0$', r'$0.1$', r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$'])

for N in sorted(co):
	axIN.errorbar(rate[N][:, 0], rate[N][:, 2], xerr=rate[N][:, 1], yerr=rate[N][:, 3],
		label=r'$N=%i$' % N, color=colors[N])
rateINsa, = axIN.plot(
	rate[_N[-1]][:, 0], vps.rate(*rate[_N[-1]][:, 0]),
	label=r'$-\inf_s \tilde{B}_{s, p^2}$', color='black', linestyle='--')
rateINa, = axIN.plot(
	rate[_N[-1]][:, 0], list(map(lambda _: mathieu.rate((np.sqrt(_), 0)), rate[_N[-1]][:, 0])),
	label=r'$I((p, 0))$', color='black')

plt.sca(axIN)
axIN.add_artist(plt.legend(loc=2,
	handles=[
		*[Line2D([0], [0], color=colors[N],
			label=r'$%s%i$' % ('N=' if N==100 else '', N))
		for N in co],
		Line2D([0], [0], color=rateINa.get_color(), linestyle=rateINa.get_linestyle(),
			label=rateINa.get_label())]))
axIN.add_artist(plt.legend(loc=4,
	handles=[
		Line2D([0], [0], lw=0,
			label=r'$D_r=%.1f$' % Dr),
		Line2D([0], [0], color=rateINsa.get_color(), linestyle=rateINsa.get_linestyle(),
			label=rateINsa.get_label())]))
figIN.subplots_adjust(left=0.25, bottom=0.20)
axIN.add_artist(plt.legend(loc=9,
	handles=[
		Line2D([0], [0], lw=0,
			label=r'$t_{\mathrm{max}}=10^2$'),
		Line2D([0], [0], lw=0,
			label=r'$N_{\mathrm{clones}}=10^3$')]))

# biased polarisation

figP, axP = plt.subplots()
axP.set_xlabel(r'$s$')
axP.set_ylabel(r'$\left<p_x\right>_s$')
axP.set_xlim([-4, 0])
axP.set_xticks([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
axP.set_xticklabels([r'$-4$', '', r'$-3$', '', r'$-2$', '', r'$-1$', '', r'$0$'])

s = np.linspace(-4, 0, 100)
dx = 1e-3

pSA, = axP.plot(s, vp.p(*s),
	label=r'$- \partial_s \max_x B_{s, p}(x)$')
pA, = axP.plot(s, mathieu.pX(*s),
	label=r'$- \partial_s \psi_p(s)$')

plt.sca(axP)
axP.add_artist(plt.legend(loc=3,
	handles=[
		Line2D([0], [0], color=line.get_color(),
			label=line.get_label())
		for line in (pSA, pA)]))
axP.add_artist(plt.legend(loc=6,
	handles=[
		Line2D([0], [0], lw=0,
			label=r'$D_r=%.1f$' % Dr)]))
figP.subplots_adjust(bottom=0.20)

# distribution coefficient

figG, axG = plt.subplots()
axG.set_xlabel(r'$s$')
axG.set_ylabel(r'$h(s)$')
axG.set_xlim([-4, 0])
axG.set_xticks([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
axG.set_xticklabels([r'$-4$', '', r'$-3$', '', r'$-2$', '', r'$-1$', '', r'$0$'])

gSA, = axG.plot(s, vp.h(*s),
	label=r'$h_p(s)$')
gA, = axG.plot(s, np.array(list(map(mathieu.optimal_potential_curvature, s)))/2,
	label=r'$\frac{1}{2} \, \partial^2_{\theta} \phi_s(\theta=0)$')

inset_axG = inset_axes(axG, loc=3, width='37.5%', height='37.5%', borderpad=1)
inset_axG.tick_params(axis='x', labelbottom=False, labeltop=True, labelsize=20)
inset_axG.tick_params(axis='y', labelleft=False, labelright=True, labelsize=20)
inset_axG.set_xlabel(r'$s$')
inset_axG.xaxis.set_label_position('top')
inset_axG.set_ylabel(r'$\mathrm{ratio}$')
inset_axG.yaxis.set_label_position('right')
inset_axG.set_xlim([-4, 0])
inset_axG.set_xticks([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
inset_axG.set_xticklabels([r'$-4$', '', '', '', '', '', '', '', r'$0$'])
inset_axG.set_ylim([1, 1.25])
inset_axG.set_yticks([1, 1.05, 1.10, 1.15, 1.20, 1.25])
inset_axG.set_yticklabels([r'$1$', '', '', '', '', r'$1.25$'])

inset_axG.plot(s, gSA.get_ydata()/gA.get_ydata())

plt.sca(axG)
axG.add_artist(plt.legend(loc=1,
	handles=[
		*[Line2D([0], [0], color=line.get_color(),
			label=line.get_label())
		for line in (gSA, gA)],
		Line2D([0], [0], lw=0,
			label=r'$D_r=%.1f$' % Dr)]))
figV.subplots_adjust(bottom=0.20)

# SAVE AND SHOW

if get_env('SAVE', default=False, vartype=bool):

	def save(fig, figname):
		fig.savefig(figname + '.eps')
		fig.savefig(figname + '.svg')

	# rate function
	save(figI, 'mathieu_rate')

	# potential
	save(figV, 'mathieu_potential')

	# rate functions from cloning
	save(figIN, 'rate_cloning')

	# biased polarisation
	save(figP, 'polarisation_x')

	# distribution coefficient
	save(figG, 'distribution_g')

if get_env('SHOW', default=True, vartype=bool): plt.show()

