import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from math import exp
from nagi.constants import EXPONENTIAL_STDP_PARAMETERS


def exp_synaptic_weight_modification(dt, a_plus, a_minus, tau_plus, tau_minus):
    if dt > 0:
        return -a_plus * exp(-dt / tau_plus)
    else:
        return a_minus * exp(dt / tau_minus)


p = EXPONENTIAL_STDP_PARAMETERS

x = [i for i in range(-40, 41)]
y = [exp_synaptic_weight_modification(dt, p['a_plus'], p['a_minus'], p['tau_plus'], p['tau_minus']) for dt in range(-40, 41)]

fig = plt.figure()
plt.ylabel("Change in synapse strength.")
plt.xlabel("Delta_t (in ms)")
plt.grid()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.plot(x, y, "r-")
plt.show()
