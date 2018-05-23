#import CNSexperiment
import ReactorTools
import numpy as np
from numpy import genfromtxt
from scipy import integrate
from array import array

import matplotlib.pyplot as plt

def flux_dEdRdP(energy, distance, power):
    """
    Computes the neutrino flux as a function of neutrino energy, 
    distance from reactor core, and the power of the reactor core

    Parameters
    ----------
    energy : float
        Neutrino energy in keV
    distance : float
        Distance in m from reactor to the point of detection
    power : float
        Reactor power in MW

    Returns
    -------
    flux : float
        Reactor neutrino flux [neutrinos/s/cm^2/keV]
    """

    return ReactorTools.dRdEnu_U235(energy) * ReactorTools.nuFlux(power, distance*100.)

def flux_dRdP(distance, power):
    """
    Computes the total neutrino flux from [1.8, 10] MeV as a function 
    of distance from reactor core and the power of the reactor core

    Parameters
    ----------
    distance : float
        Distance in m from the reactor to the point of detection
    power : float
        Reactor power in MW

    Returns
    -------
    flux : float
        Reactor neutrino flux [neutrinos/s/cm^2/keV]
    """

    return integrate.quad(flux_dEdRdP, 1800, 10000, args=(distance, power))[0] 


xmin = 1800
xmax = 10000
energies = np.linspace(xmin, xmax, 100)
u235_spectrum_300m_150MW = flux_dEdRdP(energies, 300., 150.)
fig = plt.figure(figsize=(10,8))
plt.semilogy(energies/1000,u235_spectrum_300m_150MW,'r--',label='U-235',linewidth=2)
plt.legend(prop={'size':11})
plt.xlabel('Neutrino Energy (MeV)', fontsize=14)
plt.ylabel('Flux Spectrum (nu/keV/s/cm^2)', fontsize=14)
#plt.savefig("reactor_spectrum_u235.png")
plt.show()

rmin = 5
rmax = 2000
distances = np.exp(np.linspace(np.log(rmin), np.log(rmax), 100))
u235_spectrum_150MW = np.array([flux_dRdP(dist, 150.) for dist in distances])
fig = plt.figure(figsize=(10,8))
plt.semilogx(distances, u235_spectrum_150MW, 'r--', label='U-235', linewidth=2)
plt.legend(prop={'size':11})
plt.xlabel('Distance from Reactor (m)', fontsize=14)
plt.ylabel('Total Flux (nu/s/cm^2)', fontsize=14)
plt.show()

