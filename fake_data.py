"""
Program to generate fake data for a nuclear submarine
passing a neutrino detector
"""

import numpy as np

try:
    import seaborn as sns
    seaborn_imported = True
except ImportError as e:
    print("ImportError: %s"%e)
    print("Continuing wihtout seaborn")
    seaborn_imported = False

import scipy.integrate as spint
import scipy.optimize as spopt

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
        
        def get_gaus_kernel(mus, sig):
            def gaus(x):
                kernels = [1/sig/np.sqrt(np.pi*2) * np.exp(-(x-mu)**2 / sig**2 / 2) for mu in mus]
                return np.sum(np.array(kernels))
            return gaus


        
        # Set width of individual gaussian kernels and create kde
        kde_width = 1/240
        kde = get_gaus_kernel(times, kde_width)

        # Fix binning and plot data (hist+kde) and expected rate
        histBins = 20
        norm = (tf-ti)/histBins
        timeArr = np.linspace(ti, tf, 500)
        expRate = self.count_rate(timeArr) * norm
        plt.figure(figsize=(10,8))
        plt.plot(timeArr, [kde(t)*norm for t in timeArr], label='Generated Data KDE')
        plt.hist(times, histBins, (ti, tf), label='Generated Data Histogram')
        plt.plot(timeArr, expRate, alpha = 0.8, linewidth=3, label='Expected Rate')
        plt.legend(prop={'size':14})
        plt.xlim([ti, tf])
        plt.xlabel('Time (Hours)', fontsize=15)
        plt.ylabel('Events per %.2f Hours'%norm, fontsize=15)
        plt.show()

        return np.array(times)

    def plot_count_rate(self, tl=0., th=24.):
        """
        Plot the count rate as a function of time

        Params:
            tl: Lower time to plot
            th: Upper time to plot
        """
        times = np.linspace(tl, th, 1000.)
        vec_counts = np.vectorize(self.count_rate)
        rates = vec_counts(times)
        fig = plt.figure()
        plt.plot(times, rates)
        plt.title("Closest Approach %.0f m"%self.d0)
        plt.xlabel("Time (hours)")
        plt.ylabel("Count Rate (counts/hour)")
        plt.show()
        return

    def plot_dist_vs_counts(self, rl=10., rh=5000.,
                            title_postfix=", 1,000 Detectors"):
        """
        Plot distance from reactor vs total deposited signal counts

        Params:
            rl: Lower distance of closest approach to plot
            rh: Upper distance of closest approach to plot

        Returns:
            None: Displays a plot of distance of closest approach
            vs total deposited signal counts.
        """
        d0_init = self.d0
        background_init = self.background_rate
        self.background_rate = 0.

        d0_arr = np.linspace(rl, rh)
        counts_arr = list()
        for d0 in d0_arr:
            self.d0 = d0
            # Integrate one day before and after time of closest approach
            counts_arr.append(spint.quad(self.count_rate,
                                         self.t0-24., self.t0+24.)[0])

        self.d0 = d0_init
        self.background_rate = background_init
        fig = plt.figure()
        plt.plot(d0_arr, counts_arr)
        plt.plot([rl, rh], [3., 3.], linestyle='--', color='r')
        plt.xlabel("Distance of Closest Approach (m)")
        plt.ylabel("Expected Signal Counts")
        plt.title("Closest Approach vs Total Signal" + title_postfix)
        plt.show()
        return

    def plot_detectors_vs_radius_bound(self, ndet0, ndetl=10., ndeth=100000.):
        """
        In the case of 0 background and 0 observed counts, plot the
        95% limit on the radius that can be placed as a function
        of the number of detector masses

        params:
            ndet0: Number of detectors the current setup corresponds to
            ndetl: Lower number of detectors to plot
            ndeth: Upper number of detectors to plot
        """
        d0_init = self.d0
        background_init = self.background_rate
        signal_init = self.signal_rate_0
        self.background_rate = 0.

        det_arr = np.linspace(ndetl, ndeth, 50.)
        limit_arr = list()

        def sig_counts_minus_3(r):
            self.d0 = r
            return spint.quad(self.count_rate,
                              self.t0-24., self.t0+24.)[0]-3.
        for ndet in det_arr:
            self.signal_rate_0 = signal_init*float(ndet)/float(ndet0)
            d0_guess = d0_init*float(ndet)/float(ndet0)
            temp_res = spopt.root(sig_counts_minus_3, d0_guess)
            print(temp_res)
            limit_arr.append(temp_res.x[0])
        fig = plt.figure()
        plt.plot(det_arr, limit_arr)
        plt.plot([ndetl, ndeth], [1160., 1160], linestyle='--', color='r')
        plt.plot([1380, 1380], [0., 1160], linestyle='--', color='r')
        plt.xlabel("Number of Detectors")
        plt.ylabel("95% CL Limit on Closest Approach (m)")
        plt.title("Limit For 0 Background, 0 Observed Events")
        plt.show()
        return

if __name__=="__main__":
    n_detectors = 1000. # 1000 detectors can be built for the cost of one nuclear sub
    background = 0. # Counts per hour, assume we can make a 0 background experiment
    #background = n_detectors*0.065/24. # Counts per hour, assuming KamLAND backgrounds
    # Use Double Chooz Far Detector as a reference
    signal_0 = n_detectors*66./24. # counts per hour, assuming we can build n copies of DC
    signal_dist_0 = 1050. # m
    reactor_power_0 = 6800. # MW
    t0 = 0.25 # h
    d0 = 1000. # m, worst case effective radius for a string of detectors across the Strait of Gibraltar
    v0 = 83340 # 45 knots = 83.34 m/h
    p0 = 150. # MW
    

    test_gen = FakeDataGenerator(background,
                                 signal_0, signal_dist_0,
                                 reactor_power_0,
                                 t0, d0, v0, p0)

    test_gen.plot_count_rate(0., 2*t0)
    #test_gen.plot_dist_vs_counts(rl=100., rh=1500., title_postfix=", %i Detectors"%n_detectors)
    test_gen.plot_dist_vs_counts(rl=100., rh=3000., title_postfix=", %i Detectors"%n_detectors)
    test_gen.plot_detectors_vs_radius_bound(n_detectors, ndetl=10., ndeth=2000.)

    #data = test_gen.generate(0., 2*t0)
    #print("data: %s"%data)

    #fig = plt.figure()
    #plt.hist(data, bins=20)
    #plt.show()
