#!/usr/bin/python
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
import hawkes as hwk
import numpy as np

hwk_ex = hwk.Hawkes(A=2, T=200)
nu = 1.0 / np.array([[1.5, 8], [4, 2]])
time_decay = 0.8/ np.array([0.8, 1])
baseline_rate = np.array([0.4, 0.6])

hwk_ex.set_nu(nu)
hwk_ex.set_tau(time_decay)
hwk_ex.set_baseline_rate(baseline_rate)
data = hwk_ex.simulate()

#print(data)

hwk_ex.loadData(data)
hwk_ex.plot_data_events()
hwk_ex.plot_rate_function()
print "\n* True parameters"
print "ll_train =", hwk_ex.log_likelihood(), "ll_test =", hwk_ex.log_likelihood_test_set()
print hwk_ex.test_goodness_of_fit()