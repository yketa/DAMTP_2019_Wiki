from active_work.plot import list_markers
from active_work.init import get_env

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 20})
from matplotlib.lines import Line2D

tmax = 100
g = 1
Nc = 1

I1, I2 = {}, {}

_N = [10, 20, 50, 100]
_lp = [2, 5, 10, 20, 50, 100]

markers = list_markers(_N)

# data

I1[(10, 2)] = 0.314811
I2[(10, 2)] = 0.153873

I1[(10, 5)] = 0.151473
I2[(10, 5)] = 0.0747364

I1[(10, 10)] = 0.0810729
I2[(10, 10)] = 0.0403841

I1[(10, 20)] = 0.0485856
I2[(10, 20)] = 0.0224959

I1[(10, 50)] = 0.0242837
I2[(10, 50)] = 0.0115841

I1[(10, 100)] = 0.0119906
I2[(10, 100)] = 0.00556575

I1[(20, 2)] = 0.347651
I2[(20, 2)] = 0.170548

I1[(20, 5)] = 0.172532
I2[(20, 5)] = 0.0852096

I1[(20, 10)] = 0.0883231
I2[(20, 10)] = 0.0440547

I1[(20, 20)] = 0.0554716
I2[(20, 20)] = 0.0278695

I1[(20, 50)] = 0.0227059
I2[(20, 50)] = 0.0116841

I1[(20, 100)] = 0.0101132
I2[(20, 100)] = 0.0049512

I1[(50, 2)] = 0.328074
I2[(50, 2)] = 0.163227

I1[(50, 5)] = 0.188971
I2[(50, 5)] = 0.0936609

I1[(50, 10)] = 0.0814688
I2[(50, 10)] = 0.0400027

I1[(50, 20)] = 0.0542056
I2[(50, 20)] = 0.0271517

I1[(50, 50)] = 0.0225908
I2[(50, 50)] = 0.0112064

I1[(50, 100)] = 0.0120792
I2[(50, 100)] = 0.00592474

I1[(100, 2)] = 0.341126
I2[(100, 2)] = 0.169987

I1[(100, 5)] = 0.146926
I2[(100, 5)] = 0.0727927

I1[(100, 10)] = 0.0990975
I2[(100, 10)] = 0.049325

I1[(100, 20)] = 0.0434308
I2[(100, 20)] = 0.0217006

I1[(100, 50)] = 0.0196183
I2[(100, 50)] = 0.00980965

I1[(100, 100)] = 0.0103876
I2[(100, 100)] = 0.00518348

# plot

fig, ax = plt.subplots()
ax.set_title(r'$N_c = 1, g = 1, \tau = 100$')
ax.set_xlabel(r'$\alpha$' + ' Pe')
ax.set_ylabel(r'$\frac{1}{N \mathcal{I}} - 1$')
fig.set_size_inches(12, 8)

for N in _N:
	ax.loglog(_lp, [1/I1[(N, lp)] - 1 for lp in _lp], color='blue', marker=markers[N])
	ax.loglog(_lp, [1/I2[(N, lp)] - 1 for lp in _lp], color='red', marker=markers[N])

ax.plot(_lp, _lp, color='black', lw=4)

legend = [Line2D([0], [0], color='blue',
	label=r'$\mathcal{I}_1(0, \tau)$')]
legend += [Line2D([0], [0], color='red',
	label=r'$\mathcal{I}_2(0, \tau)$')]
for N in _N:
	legend += [Line2D([0], [0], color='black', lw=0, marker=markers[N],
		label=r'$N=%i$' % N)]
legend += [Line2D([0], [0], color='black', lw=4,
	label=r'$\mathcal{I}_{1, th.}$')]
ax.legend(handles=legend, prop={'size': 20})

if get_env('SAVE', default=False, vartype=bool):
	fig.savefig('testI1.eps')
	fig.savefig('testI1.svg')

if get_env('SHOW', default=True, vartype=bool): plt.show()

