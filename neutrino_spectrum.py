#import CNSexperiment
import ReactorTools
import numpy as np
from numpy import genfromtxt

from array import array

import matplotlib.pyplot as plt

xmin = 1800
xmax = 10000
energies = np.linspace(xmin, xmax)
u235_spectrum = ReactorTools.dRdEnu_U235(energies)
fig = plt.figure()
plt.semilogy(energies,u235_spectrum,'r--',label='U-235',linewidth=2)
plt.legend(prop={'size':11})
plt.xlabel('Neutrino Energy (keV)', fontsize=14)
plt.ylabel('Spectrum (nu/keV/fission)', fontsize=14)
plt.savefig("reactor_spectrum_u235.png")
plt.show()

