from active_work.init import get_env
from active_work.plot import list_colormap
from active_work.miscellaneous import RTPring
r = RTPring()

import matplotlib.pyplot as plt
try: plt.style.use('paper')
except: print('Matplotlib stylesheet \'paper\' does not exist.')

import numpy as np

Lambda = np.linspace(-50, 2, 100)

# Psi

figL, axL = plt.subplots()
axL.set_xlabel(r'$\Lambda$')
axL.set_xlim([-50, 10])
axL.set_xticks([-50, -40, -30, -20, -10, 0, 10])
axL.set_xticklabels(['', r'$-40$', '', r'$-20$', '', r'$0$', ''])
axL.set_ylabel(r'$\Psi(\Lambda)$')
axL.set_ylim([-np.pi**2/4, np.pi**2/8])
axL.set_yticks([-np.pi**2/4, -3*np.pi**2/16, -np.pi**2/8, -np.pi**2/16, 0, np.pi**2/16, np.pi**2/8])
axL.set_yticklabels([r'$-\frac{\pi^2}{4}$', r'$-\frac{3 \pi^2}{16}$', r'$-\frac{\pi^2}{8}$', r'$-\frac{\pi^2}{16}$', r'$0$', r'$\frac{\pi^2}{16}$', r'$\frac{\pi^2}{8}$'])

axL.plot(Lambda, list(map(r.Psi, Lambda)))

axL.axhline(y=0, color='black', linestyle='-', lw=2)
axL.axvline(x=0, color='black', linestyle='-', lw=2)

adjust = {'left': 0.25, 'top': 0.92, 'right': 0.99}
figL.subplots_adjust(**adjust)

# Gamma

figG, axG = plt.subplots()
axG.set_xlabel(r'$\Lambda$')
axG.set_xlim([-50, 10])
axG.set_xticks([-50, -40, -30, -20, -10, 0, 10]) 
axG.set_xticklabels(['', r'$-40$', '', r'$-20$', '', r'$0$', ''])
axG.set_ylabel(r'$\Psi(\Lambda)/\Lambda$')
axG.set_ylim([0.0, 1.6])

axG.plot(Lambda, list(map(r.Gamma, Lambda)))

axG.axhline(y=1, color='black', linestyle='-', lw=2)
axG.axvline(x=0, color='black', linestyle='-', lw=2)

figG.subplots_adjust(**adjust)

# epsilon

fige, axe = plt.subplots()
axe.set_xlabel(r'$r/L$')
axe.set_ylabel(r'$L \varepsilon_{\Lambda}(r)$')

x = np.linspace(0, 1, 100)
_Lambda = [-3, -2, -1, 0, 1, 2]
colors = list_colormap(_Lambda, sort=True)
for l in sorted(_Lambda):
	axe.plot(x, r.LEpsilon(l, *x), color=colors[l],
		label=r'$\Lambda=%i$' % l)

plt.sca(axe)
plt.legend(loc='upper center', ncol=3)

fige.set_size_inches(11.25, 5)
fige.subplots_adjust(left=0.12, top=0.98, right=0.97, bottom=0.21)

# show/save

if get_env('SAVE', default=False, vartype=bool):

	def save(f, fname):
		f.savefig(fname + '.eps')
		f.savefig(fname + '.svg')

	save(figL, 'lambdaRTP')
	save(figG, 'gammaRTP')
	save(fige, 'epsilonRTP')

if get_env('SHOW', default=True, vartype=bool): plt.show()

