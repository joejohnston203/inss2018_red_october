"""
Program to generate fake data for a nuclear submarine
passing a neutrino detector
"""

import numpy as np

import scipy.integrate as spint

import matplotlib.pyplot as plt

class FakeDataGenerator:
    def __init__(self, background_rate,
                 signal_rate_0, signal_distance_0,
                 reactor_power_0,
                 t0, d0, v0, p0):
        """
        Initialize the generator with

        Params:
            background: Background rate in counts per hour
            signal_rate_0: Signal rate in counts per
                hour at the given reference distance
            signal_distance_0: Distance in m at which
                the given signal rate is measured
            reactor_power_0: Reactor power at which the given
                signal rate is measured, in MW
            t0: Time at which the submarine passes most closely
                to the detector, in hours
            d0: Distance of closest approach to the detector, in m
            v0: Velocity of the sub, in m/hour
            p0: Power of the sub in MW

        Returns:
            None
        """
        self.background_rate = background_rate
        self.signal_rate_0 = signal_rate_0
        self.signal_distance_0 = signal_distance_0
        self.reactor_power_0 = reactor_power_0
        self.t0 = t0
        self.d0 = d0
        self.v0 = v0
        self.p0 = p0
        pass

    def count_rate(self, t):
        """
        Total count rate as a function of time

        Params:
            t: Time in hours

        Returns:
            float: Count rate in counts/hour
        """
        r = np.sqrt(self.d0**2+
                    ((t-self.t0)*self.v0)**2)
        return self.background_rate + \
            (self.signal_rate_0*(self.signal_distance_0/r)**2*\
            (self.p0/self.reactor_power_0))

    def generate(self, ti=0., tf=24.):
        """
        Generate fake data, saves it in the object, then returns
        the data.

        params:
            ti, tf: Initial and final times for time window in which
                data is generated, in hours

        Returns:
            np.array(float): Array of times at which a count is
            detected, in hours
        """
        exp_counts = spint.quad(self.count_rate, ti, tf)[0]
        print("Total expected counts: %.2e"%exp_counts)
        meas_counts = np.random.poisson(exp_counts)
        print("Measured counts: %i"%meas_counts)
        
        # We know the maximum count rate occurs at t=t0
        cr_max = self.count_rate(self.t0)
        times = list()
        while len(times)<meas_counts:
            t = np.random.uniform(ti, tf)
            cr_curr = self.count_rate(t)
            x = np.random.uniform()
            if x<cr_curr/cr_max:
                times.append(t)
        return np.array(times)

if __name__=="__main__":
    background = 0. # Counts per hour
    # Use Double Chooz Far Detector as approximation
    #signal_0 = 1000./24.# counts per hour
    signal_0 = 1000000.# counts per hour, note that this is a huge signal rate for testing purposes
    signal_dist_0 = 1000. # m
    reactor_power_0 = 8500. # MW
    t0 = 1. # h
    d0 = 1000. # m
    v0 = 83340 # 45 knots = 83.34 m/h
    p0 = 150. # MW
    
    test_gen = FakeDataGenerator(background,
                                 signal_0, signal_dist_0,
                                 reactor_power_0,
                                 t0, d0, v0, p0)
    data = test_gen.generate(0., 2.)
    #print("data: %s"%data)

    fig = plt.figure()
    plt.hist(data, bins=20)
    plt.show()
        
