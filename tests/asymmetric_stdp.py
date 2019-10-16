import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from nagi.constants import ASYMMETRIC_HEBBIAN_PARAMS
from nagi.stdp import *


def plot_stdp(f):
    p = ASYMMETRIC_HEBBIAN_PARAMS

    x = [i for i in range(-40, 41)]
    y = [f(dt, p['a_plus'], p['a_minus'], p['tau_plus'], p['tau_minus']) for dt in range(-40, 41)]

    fig = plt.figure()
    plt.ylabel("delta_w")
    plt.xlabel("delta_t (in ms)")
    plt.grid()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.plot(x, y, "r-")
    plt.title(f.__name__)
    plt.show()


if __name__ == '__main__':
    plot_stdp(asymmetric_hebbian)
    plot_stdp(asymmetric_anti_hebbian)
