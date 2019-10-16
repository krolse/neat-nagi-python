import matplotlib.pyplot as plt
from nagi import snn, constants
from nagi.constants import TIME_STEP_IN_MSEC


def plot_spikes(spikes, title):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f, w in spikes]
    v_values = [v for t, I, v, u, f, w in spikes]
    u_values = [u for t, I, v, u, f, w in spikes]
    I_values = [I for t, I, v, u, f, w in spikes]
    f_values = [f for t, I, v, u, f, w in spikes]
    w_values = [w for t, I, v, u, f, w in spikes]

    fig = plt.figure()
    plt.subplot(5, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(5, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(5, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "m-")

    plt.subplot(5, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "c-")

    plt.subplot(5, 1, 5)
    plt.ylabel("Weight")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, w_values, "b-")

    fig = plt.figure()
    plt.title("Izhikevich's spiking neuron model u/v ({0!s})".format(title))
    plt.xlabel("Recovery (u)")
    plt.ylabel("Potential (mv)")
    plt.grid()
    plt.plot(u_values, v_values, 'r-')

    plt.show()
    plt.close()


def show(title, a, b, c, d):
    neuron = snn.SpikingNeuron(5, a, b, c, d, {0: 0.5})
    network = snn.SpikingNeuralNetwork({1: neuron}, [0], [1])
    spike_train = []
    for i in range(5000):
        network.set_inputs([40 if i % 50 == 0 else 0])
        spike_train.append((TIME_STEP_IN_MSEC * i, neuron.current, neuron.membrane_potential, neuron.membrane_recovery, neuron.fired,
                            neuron.inputs[0]))
        network.advance(TIME_STEP_IN_MSEC)

    plot_spikes(spike_train, title)


if __name__ == '__main__':
    show('regular spiking', **constants.REGULAR_SPIKING_PARAMS)
    show('intrinsically bursting', **constants.INTRINSICALLY_BURSTING_PARAMS)
    show('chattering', **constants.CHATTERING_PARAMS)
    show('fast spiking', **constants.FAST_SPIKING_PARAMS)
    show('low-threshold spiking', **constants.LOW_THRESHOLD_SPIKING_PARAMS)
    show('thalamo-cortical', **constants.THALAMO_CORTICAL_PARAMS)
    show('resonator', **constants.RESONATOR_PARAMS)

    plt.show()
