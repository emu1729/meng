#!/usr/bin/python

# multivariate Hawkes process -- simulation and Bayesian inference
# Richard Kwo
# May, 2014

from numpy import zeros, ones, identity, bincount, log, exp, abs, sqrt, savez, savetxt, shape, eye, all, any, argmin, argmax, array, mean, linspace, sum, loadtxt, concatenate, amax, diag
from slice_sampler import slice_sample
from numpy.random import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.special import gammaln

from likelihoods import log_gamma

from scipy.optimize import minimize


# class begin
class Hawkes_Inf_IRM(object):
    """docstring for Hawkes"""
    def __init__(self, A=4, T=100.0, C=2, non_diagonal=False):
        '''
        Hawkes object
        A: number of agents
        T: max time
        C: number of groups
        non_diagonal: if True, the diagonal of the nu matrix is enforced to zero 
        '''
        assert A>=1

        self._A = A
        self._T = T * 1.
        self._C = C
        self._non_diagonal = non_diagonal

        print("Hawkes initialized (A=%d, T=%.2f, C=%d, non_diagonal=%s)" % (self._A, self._T, self._C, self._non_diagonal))

        # parameters
        self._clusters_assign = zeros((self._A))
        self._alpha = None      
        self._nu = zeros((self._C, self._C))
        self._gamma = zeros((self._C))
        self._kappa = zeros((self._A))
        self._tau = zeros(self._C)
        self._baseline_rate = zeros((self._C, self._C))

        # hyperparams
        self._hyperparams_tau = None    # gamma (shape, scale) hyperparams
        self._clusters_set = False
        self._nu_set = False
        self._tau_set = False
        self._kappa_set = False
        self._baseline_rate_set = False

        # data 
        self._data = None

        # locks
        self._ll_cached = False
        self._prior_cached = False
        self._log_likelihood = None
        self._log_likelihood_test_set = None
        self._log_prior = None

    def get_A(self):
        ''' get A '''
        return self._A

    def get_T(self):
        ''' get max time T'''
        return self._T

    def set_kappa(self, marks, f, x=None, y=None):
        ''' set influence parameters
        marks is a matrix of size A x M, M represents dimension of marks
        f is a function that operates on vectors of size M
        '''
        assert marks.shape[0] == self._A
        for a in range(marks.shape[0]):
            self._kappa[a] = f(marks[a])
        self._kappa_set = True

    def set_cluster_assign(self, clusters):
        ''' set cluster assignments
        '''
        assert clusters.size == self._A
        assert np.amax(clusters) == self._C - 1
        assert np.amin(clusters) == 0
        self._clusters_assign = clusters
        self._clusters_set = True

    def set_nu(self):
        '''set the jump height matrix self._nu[x,y] to nu.
        When setting the whole self._nu to an A x A matrix nu, passing x = None and y = None
        nu[p, q] is the excitation height from p to q
        '''
        assert self._kappa_set
        assert self._clusters_set
        sums = zeros(self._C)
        for a in range(self._A):
            sums[self._clusters_assign[a]] += self._kappa[a]
        for i in range(self._C):
            for j in range(self._C):
                self._nu[i][j] = sums[i]*sums[j]
                if self._non_diagonal:
                    if i == j:
                        self._nu[i][j] = 0
        self._nu_set = True

    def set_tau(self, tau, x=None):
        '''set the time decays, of shape (C)'''
        if all(self._tau[x] == tau): return  # avoid reevaluating ll when not changed
        if x is None:
            assert shape(tau) == (self._C, ) and all(tau>0), '%s' % (tau)
        else:
            assert tau>0
        self._tau[x] = tau * 1.0
        self._tau_set = True


    def set_baseline_rate(self, gamma):
        '''set the baseline_rate (gamma n_p n_q), of C x C'''
        assert self._clusters_set
        assert gamma.size == self._C
        unique, counts = np.unique(self._clusters_assign, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        self._gamma = gamma
        vals = list(cluster_counts.values())
        for i in range(self._C):
            for j in range(self._C):
                self._baseline_rate[i][j] = self._gamma[i] * vals[i] * vals[j]
        self._baseline_rate_set = True

    def get_nu(self):
        return self._nu.copy()

    def get_tau(self):
        return self._tau.copy()

    def get_baseline_rate(self):
        return self._baseline_rate.copy()

    def get_kappa(self):
        return self._kappa.copy()

    def test_stationarity(self):
        time_decay_mat = zeros((self._C, self._C))
        for x in range(self._C):
            time_decay_mat[:,x] = self._tau
        gamma_mat = self._nu * time_decay_mat
        w, v = np.linalg.eig(gamma_mat)
        if max(abs(w))<1:
            return True
        else:
            # print "Not stationary: sprectrum is", w
            return False

    def test_matrix_norm_constraint(self):
        '''
        Test if the matrix norm constraint (1->1 matrix norm < 1, 
            i.e. all the col sum of the matrix[x,y] = nu[x,y] * tau[y] < 1) is satisfied. 
        
        This is a stronger condition to ensure stationarity.
        '''
        colsum = sum(self._nu, axis=0) * self._tau
        if all(colsum<1): 
            return True
        else:
            return False


    def plot_data_events(self, show_figure=True):
        '''
        Show a vertical line plot of data.
        '''
        assert self._data is not None

        fig = plt.figure()
        for x in range(self._A):
            ax = fig.add_subplot(self._A,1,x+1)
            ax.vlines(x=self._data[x], ymin=0, ymax=1)
            ax.axvline(x=self._time_to_split, ls='--', color='r')
            ax.set_title("agent %d (%d events)" % (x+1, len(self._data[x])))
            ax.set_xlim(0, self._T)
            ax.set_ylim(0, 1)

        plt.tight_layout(pad=1, h_pad=1, w_pad=1)
        if show_figure:
            plt.show()


    def simulate(self):
        '''simulate a sample path of the multivariate Hawkes process 
        method: Dassios, A., & Zhao, H. (2013). Exact simulation of Hawkes process with exponentially decaying intensity. Electronic Communications in Probability
        ''' 
        assert self._nu_set and self._tau_set and self._baseline_rate_set, "parameters not set yet"
        assert self.test_stationarity(), "This is not a stationary process!"

        currentTime = 0
        # data[x] is the time points for x
        data = [[] for x in xrange(self._C * self._C)]
        # the left and right limit of intensities
        intensities_left = self._baseline_rate.copy().flatten()
        intensities_right = self._baseline_rate.copy().flatten() + 1e-10
        # event counts
        event_counts = [0 for x in xrange(self._C * self._C)]

        print("Simulating...")
        iter_count = 0
        while currentTime <= self._T:
            iter_count += 1
            # get W
            s = -ones(self._C * self._C)
            for x in range(self._C * self._C):
                u1 = random()
                D = 1 + log(u1)/(intensities_right[x] - self._baseline_rate.flatten()[x])/self._tau[x]
                u2 = random()
                s2 = -1/self._baseline_rate[x] * log(u2)
                if D>0:
                    s1 = -self._tau[x] * log(D)
                    s[x] = min(s1, s2)
                else:
                    s[x] = s2
            assert all(s>=0)
            # event to l 
            l = argmin(s)
            W = s[l]

            # record jump
            currentTime = currentTime + W
            data[l].append(currentTime)
            event_counts[l] += 1

            # update intensities
            intensities_left = (intensities_right - self._baseline_rate) * exp(-W / self._tau) + self._baseline_rate
            intensities_right = intensities_left + self._nu[l,:]

        print("Simulation done, %d events occurred" % (sum(event_counts)))

        return data

# class over

def unitest_simulation():
    A = 4
    T = 200.0
    C = 2
    print("A = ", A)
    print("max time T = ", T)
    print("C = ", C)
    hawkes_proc = Hawkes_Inf_IRM(A=A, T=T, C=C)

    # parameters
    # define simple linear function for f
    def f(mark):
        return 0.1*mark[0]
    marks = np.array([[1], [1], [2], [2]])
    hawkes_proc.set_kappa(marks, f)
    print("Kappa = ", hawkes_proc.get_kappa())
    cluster_assign = np.array([0, 1, 0, 1])
    hawkes_proc.set_cluster_assign(cluster_assign)
    hawkes_proc.set_nu()
    print("Nu = ", hawkes_proc.get_nu())
    time_decay = 0.8/ array([0.8, 1])
    hawkes_proc.set_tau(time_decay)
    gamma = np.array([0.1, 0.2])
    hawkes_proc.set_baseline_rate(gamma)
    print("Baseline_rate = ", hawkes_proc.get_baseline_rate())

    print("stationarity:", hawkes_proc.test_stationarity())
    print("strong stationarity (matrix norm constraint): ", hawkes_proc.test_matrix_norm_constraint())

    '''
    # simulate
    data = hawkes_proc.simulate()
    '''


if __name__ == '__main__':
    unitest_simulation()