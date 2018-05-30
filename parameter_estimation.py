#import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import integrate
from scipy.stats import poisson
import os
import emcee
import prior

class AnalyzeAntiNueCounts:
    """
    Class for analyzing neutrino counts for generated data of a nuclear submarine
    going by

    Instantiator Args:
        ndim: number of parameters for model
        t (array): times of neutrino detection
    """

    def __init__(self, t, background, signal, tMax=24.):
        """
        Initialize global variables of ship passing model
    
        Args:
            t (array): times of neutrino detection
            tMax (double): maximum time of simulaton
            expBackground (double): average background rate of neutrino detection
        """

        # Set global variables
        self.ndim = 2
        self.t = t
        self.background = background
        self.signal = signal
        self.tMax = tMax
        
        # Create an array of ndim Priors
        self.priors = [ prior.Prior(0,1) for i in range(self.ndim) ]


    def set_priors(self, prior_type, param1, param2):
        """
        Setter for the priors on all model parameter

        Args:
            prior_type (array): type of prior to set (uniform, gaussian, jefferys)
            param1 (array): 1st parameter of prior (either lower bound or mean)
            param2 (array): 2nd parameter of prior (either upper bound or std dev)
        """

        # Assign Prior object depending on requested type 
        for i in range(self.ndim):
            if (prior_type[i] == 'uniform'):
                self.priors[i] = prior.LogUniformPrior(param1[i], param2[i])
            elif (prior_type[i] == 'gaussian'):
                self.priors[i] = prior.LogGaussianPrior(param1[i], param2[i])
            elif (prior_type[i] == 'jefferys'):
                self.priors[i] = prior.LogJefferysPrior(param1[i], param2[i])
            elif (prior_type[i] == 'poisson'):
                self.priors[i] = prior.LogPoissonPrior(param1[i])
            elif (prior_type[i] == 'exponentialdecay'):
                self.priors[i] = prior.LogExponentialDecayPrior(param1[i])
            else:
                print("Invalid prior option. Modify inputs and try again.")


    def log_prior(self, params):
        """
        Computes log of prior probability distribution

        Args:
            params (array): Model parameters

        Returns:
            log of prior of all parameters
        """
        priors_eval = [ self.priors[i](params[i]) for i in range(self.ndim) ]
        
        return np.sum(priors_eval)    

    
    def log_post(self, params):
        """
        Returns log of posterior probability distribution for traditional climate model

        Args:
            params (array): Parameters for the traditional (simple) climate model
       
        Returns:
            Log of posterior distribution
        """

        return self.log_prior(params) + self.log_lh(params)


    def __call__(self, params):
        """
        Evaluate the model for given params

        Args:
            params: parameters from MCMC
        
        Returns:
             rate
        """

        x_model = np.linspace(0, self.tMax, 500)
        y_model = np.array([self.rate(t, params) for t in x_model])

        return x_model, y_model

    
    def rate(self, t, params):
        """
        Evaluate the expected event rate at a time t

        Args:
            params: input parameters of model (t0, d0)
            t (double): time at which to evaluate the rate
        """

        # Time and distance of closest approach of submarine to detector
        t0, d0 = params

        # Base distance of model detector to reactor (m)
        baseDistance = 1050
        # Base power of reactor (MW)
        basePower = 6800
    
        # Submarine velocity (m/hr)
        subVelocity = 83340
        # Submarine power (MW)
        subPower = 150

        subDist = 1/(d0**2 + subVelocity**2*(t-t0)**2)
        expRate = self.background + (self.signal*subPower/basePower*baseDistance**2) * subDist

        return expRate

    
    def log_lh(self, params):
        """
        Computes log of Gaussian likelihood function

        Args:
            params (array): Parameters for the model,
            contain subset (in order) of the following parameters:
                -t0: time of closest approach of the submarine to the detector
                -d0: distance of closest approach of the submarine to the detector

        Returns:
            log likelihood  
        """

        def rate(x):
            return self.rate(x, params)

        NexpTot = int(integrate.quad(rate, 0, self.tMax)[0])
        logL = np.log(poisson.pmf(len(self.t), NexpTot))
        for i in range(len(self.t)):
            logL += np.log(rate(self.t[i]))
        return logL



    def run_MCMC(self, param_guess, nwalkers, nsteps):
        """
        Samples the posterior distribution via the affine-invariant ensemble 
        sampling algorithm; plots are output to diagnose burn-in time; best-fit
        parameters are printed; best-fit line is overplotted on data, with errors.
    
        Args:
            param_guess (array): Initial guess for parameters. Can have length < 5
            nwalkers (int): Number of walkers for affine-invariant ensemble sampling;
                            must be an even number
            nsteps (int): Number of timesteps for which to run the algorithm


        Returns:
            Samples (array): Trajectories of the walkers through parameter spaces.
                             This array has dimension (nwalkers) x (nsteps) x (ndim)
        """
        
        # Set walker starting locations randomly around initial guess
        starting_positions = [
            np.array(param_guess) + 1e-2 * np.random.randn(self.ndim) for i in range(nwalkers)
        ]

        # Set up the sampler object
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_post, args=())

        # Progress bar
        width = 100
        for i, result in enumerate(sampler.sample(starting_positions, iterations=nsteps)):
            n = int((width+1) * float(i) / nsteps)
            if (i == 0):
                print('Progress: ')
                
            print("\r[{0}{1}]".format('#' * n, ' ' * (width - n)), end='')

        # return the samples for later output
        self.samples = sampler.flatchain
        self.sampler = sampler
        return self.samples


    def show_results(self, burnin):
        """
        Displays results from self.sample
    
        Args:
            burnin (int): Burn in time to trim the samples
        """  
        
        # Modify self.samples for burn-in
        self.samples = self.sampler.chain[:, burnin:, :].reshape((-1, self.ndim))

        # Get number walkers and number of iterations
        nwalkers = self.sampler.chain.shape[0]
        nit = self.sampler.chain.shape[1]

        # Plot the traces and marginalized distributions for desired parameters
        fig, ax = plt.subplots(2*self.ndim,
                               figsize=(10,self.ndim*3.))
        plt.subplots_adjust(hspace=0.5)

        for i in range(self.ndim):
            ax[2*i].set(ylabel="Parameter %d"%i)
            ax[2*i+1].set(ylabel="Parameter %d"%i)
            sns.distplot(self.samples[:,i], ax=ax[2*i])

            for j in range(nwalkers):
                ax[2*i+1].plot(np.linspace(1+burnin,nit,nit-burnin), self.sampler.chain[j,burnin:,i], "b")
        plt.show()

        # Store the samples in a dataframe
        index = [i for i in range(len(self.samples[:,0]))]
        columns = ['p'+str(i) for i in range(self.ndim)]
        samples_df = pd.DataFrame(self.samples, index=index, columns=columns)

        # Compute and print the MAP values
        q = samples_df.quantile([0.16, 0.50, 0.84], axis=0)
        for i in range(self.ndim):
            print("Param {:.0f} = {:.6f} + {:.6f} - {:.6f}".format( i, 
            q['p'+str(i)][0.50], q['p'+str(i)][0.84] - q['p'+str(i)][0.50], q['p'+str(i)][0.50] - q['p'+str(i)][0.16]))

        # Best-fit params
        self.bestfit_params = [ q['p'+str(i)][0.50] for i in range(self.ndim) ]
        
        # Evaluate the model with best fit params
        x_model, y_model = self.__call__(self.bestfit_params)
        
        # Plot the best-fit line, and data
        plt.figure(figsize=(14,8))
        #sns.distplot(self.t, hist=False, label='Data KDE')
        plt.plot(x_model, y_model, label='Modle Best Fit')
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('Model Prediction', fontsize=12)
        plt.xlim([0, self.tMax])
        plt.title('Model Fit to Data');
        plt.legend()
        plt.show()


    def get_samples(self):
        """
        Getter that returns the samples
        """

        return self.samples

        
    def set_parameters(self, params):
        """
        Setter for model parameters

        Args:
            params (array): parameters to set
        """

        self.bestfit_params = params

    
    def get_parameters(self):
        """
        Getter for model parameters
        """

        return self.bestfit_params
