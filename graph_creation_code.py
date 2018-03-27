import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.array([2.0, 18.0, 19.0, 20.0])
y = x*1.2
yerr = np.array([1.0, 1.2, 1.4, 1.5])
xerr = np.array([0.0, 0.0, 0.0, 0.0])
ls = 'dotted'

fig, ax = plt.subplots(figsize=(20, 13))

# do the plotting
ax.errorbar(x, y, yerr=yerr, xerr=xerr,
            fmt='rs--',
            marker='.', markersize=12,
            linestyle='none',
            capsize=5,  # cap length for error bar
            capthick=0.5  # cap thickness for error bar
            )

x = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
y = x*1.2
yerr = np.array([0.5, 3.2, 4.4, 5.5, 5.6])
xerr = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
ls = 'dotted'

ax.errorbar(x, y, yerr=yerr, xerr=xerr, uplims=True,
            fmt='*',
            marker='.', markersize=12,
            linestyle='none',
            capsize=5,  # cap length for error bar
            capthick=0.5  # cap thickness for error bar
            )

# tidy up the figure
ax.set_xlim((0, 21.0))
ax.set_title('Errorbar upper and lower limits')
plt.show()