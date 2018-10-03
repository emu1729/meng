## Module to import data and run hawkes BEC model on select data
import numpy as np

## Import relevant data
import MIDIP_import as MIDIP
countries = MIDIP.red_countries
time_seq = MIDIP.red_time_seq

## Import the BEC model
import hawkes as hawkes

## Set relevant parameters
def maxTime(data):
	max_all = 0
	for data_val in data:
		time = max(data_val)
		if time > max_all:
			max_all = time
	return max_all

A = len(countries)
T = maxTime(time_seq)

print("A = ", A)
print("max time T = ", T)
hawkes_proc = hawkes.Hawkes(A=A, T=T)
hawkes_proc.loadData(time_seq, time_to_split=T*0.5)
hawkes_proc.plot_data_events()

# MLE estimate
MLE_baseline_rate, MLE_tau, MLE_nu = hawkes_proc.optimize_parameters()
print("\n* MLE")
print("baseline rate: ", "\nMLE =", MLE_baseline_rate)
print("time decay: ", "\nMLE =", MLE_tau)
print("influence matrix: ", "\nMLE =", MLE_nu)
print("ll_train =", hawkes_proc.log_likelihood(), "ll_test =", hawkes_proc.log_likelihood_test_set())  
hawkes_proc.test_goodness_of_fit()

# MCMC
B = 100
N = 100
## containers
nu_samples = np.zeros((N, A, A))
tau_samples = np.zeros((N, A))
baseline_rate_samples = np.zeros((N, A))
ll_samples = np.zeros(N)
ll_test_samples = np.zeros(N)
## sample
for k in range(B+N):
	if k%20==0: print("Sampling %d out of %d ..." % (k, B+N))
	if k>=B:
		nu_samples[k-B, ...] = hawkes_proc.sample_nu()
		tau_samples[k-B, ...] = hawkes_proc.sample_tau()
		baseline_rate_samples[k-B, ...] = hawkes_proc.sample_baseline_rate()
		ll_samples[k-B] = hawkes_proc.log_likelihood()
		ll_test_samples[k-B] = hawkes_proc.log_likelihood_test_set()

## plot
print("\n* MCMC")
print("ll_train =", mean(ll_samples), "ll_test =", mean(ll_test_samples))
print("influence matrix")
print("sample mean:", mean(nu_samples, axis=0))
print("")
print("time decay")
print("sample mean:", mean(tau_samples, axis=0))
print("")
print("baseline rate")
print("sample mean: ", mean(baseline_rate_samples, axis=0))

hawkes_proc.set_nu(mean(nu_samples, axis=0))
hawkes_proc.set_tau(mean(tau_samples, axis=0))
hawkes_proc.set_baseline_rate(mean(baseline_rate_samples, axis=0))

hawkes_proc.test_goodness_of_fit()