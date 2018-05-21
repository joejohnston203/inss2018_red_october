Python scripts used to calculate the neutrino spectrum with equations from
P. Huber, Phys.Rev. C84, 024617 (2011), arXiv:1106.0687 [hep-ph].

ReactorTools.py calculates the spectrum in units of neutrinos/keV/fission. The "_corrected" methods use the corrected data files rather than the polynomial fit.

neutrino_spectrum.py plots the U-235 spectrum.

The spectrum can be multiplied by a normalization factor get the flux in neutrinos/keV/s/cm^2. For example, assuming a reactor power of 8.54 GW and a reactor distance of 400 m, the Double Chooz normalization factor is:

8540 MW/200 MeV per fission /1.602*10^-19 MW per MeV / (4*pi*(40000 m)^2)
= 1.326*10^10 fissions/s/cm^2

