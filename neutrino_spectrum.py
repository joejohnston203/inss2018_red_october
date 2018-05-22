#import CNSexperiment
import ReactorTools
import numpy as np
from numpy import genfromtxt

from array import array

import matplotlib.pyplot as plt

xmin = 1800
xmax = 20000
energies = np.linspace(xmin, xmax)
u235_spectrum = ReactorTools.dRdEnu_U235(energies)
fig = plt.figure(figsize=(10,8))
plt.semilogy(energies/1000,u235_spectrum,'r--',label='U-235',linewidth=2)
plt.legend(prop={'size':11})
plt.xlabel('Neutrino Energy (MeV)', fontsize=14)
plt.ylabel('Flux Spectrum (nu/keV/fission)', fontsize=14)
#plt.savefig("reactor_spectrum_u235.png")
plt.show()

