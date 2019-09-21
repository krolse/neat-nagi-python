TIME_STEP_IN_MSEC = 0.05
MEMBRANE_POTENTIAL_THRESHOLD = 30.0

REGULAR_SPIKING_PARAMS = {'a': 0.02, 'b': 0.20, 'c': -65.0, 'd': 8.00}
INTRINSICALLY_BURSTING_PARAMS = {'a': 0.02, 'b': 0.20, 'c': -55.0, 'd': 4.00}
CHATTERING_PARAMS = {'a': 0.02, 'b': 0.20, 'c': -50.0, 'd': 2.00}
FAST_SPIKING_PARAMS = {'a': 0.10, 'b': 0.20, 'c': -65.0, 'd': 2.00}
THALAMO_CORTICAL_PARAMS = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05}
RESONATOR_PARAMS = {'a': 0.10, 'b': 0.25, 'c': -65.0, 'd': 2.00}
LOW_THRESHOLD_SPIKING_PARAMS = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.00}

ASYMMETRIC_HEBBIAN_PARAMS = {'a_plus': 1, 'a_minus': 1, 'tau_plus': 10, 'tau_minus': 10, 'sigma': 0.1, 'w_max': 1.5, 'w_min': -1.5}
STDP_PARAMS = {'sigma': 0.1, 'w_max': 1.5, 'w_min': -1.5}
