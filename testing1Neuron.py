

import brian2 as b2
from neurodynex.tools import input_factory, plot_tools
import random
import matplotlib.pyplot as plt
import numpy as np
# Neuron model default values
MEMBRANE_TIME_SCALE = 1. * b2.ms
G_LEAK = 1000 * b2.msiemens
V_LEAK = -70 * b2.mV
C_M = 50.0 * b2.mfarad
R_POT = -70 * b2.mV
MEMBRANE_RESISTANCE = 10. * b2.Mohm
FIRING_THRESHOLD = -100 * b2.mV
def simulate_1_WORM_neuron(input_current,
                        simulation_time=5 * b2.ms,
                        v_leak = V_LEAK,
                        g_leak = G_LEAK,
                        c_m = C_M,
                        rest_pot = R_POT,
                        tau=MEMBRANE_TIME_SCALE,
                        m_res=MEMBRANE_RESISTANCE
                         ):


    # differential equation of neuron model
    eqs = """
    dv/dt =
    ( g_leak * (v_leak - v) + input_current(t,i)  ) / c_m : volt 
    """



    # LIF neuron using Brian2 library
    neuron = b2.NeuronGroup(
        1, model=eqs, method="linear")
    neuron.v = rest_pot  # set initial value

    # monitoring membrane potential of neuron and injecting current
    state_monitor = b2.StateMonitor(neuron, ["v"], record=True)
    # run the simulation
    b2.run(simulation_time)
    return state_monitor



def simulate_WORM_neuron(input_current,
                        simulation_time=5 * b2.ms,
                        v_leak = V_LEAK,
                        g_leak = G_LEAK,
                        c_m = C_M,
                        rest_pot = R_POT,
                        tau=MEMBRANE_TIME_SCALE,
                        f_t=FIRING_THRESHOLD
                         ):


    # differential equation of neuron model
    eqs = """
    dv/dt =
    ( g_leak * (v_leak - v) + input_current(t,i)  ) / c_m : volt 
    """



    # LIF neuron using Brian2 library
    neuron = b2.NeuronGroup(
        2, model=eqs,threshold='v>f_t', method="linear")
    neuron.v = rest_pot  # set initial value

    # monitoring membrane potential of neuron and injecting current
    state_monitor = b2.StateMonitor(neuron, ["v"], record=True)
    S=b2.Synapses(neuron,neuron,model='w : volt',on_pre='v += w')
    S.connect(i=0, j=1)
    S.w = 0.01*b2.mV
    # run the simulation
    b2.run(simulation_time)
    return state_monitor

def only_input_step_current(t_start, t_end, unit_time, amplitude, append_zero=True):

    tmp_size = 1 + t_end  # +1 for t=0
    if append_zero:
        tmp_size += 1
    tmp = np.zeros((tmp_size, 2)) * b2.amp
    tmp[t_start: t_end + 1, 0] = amplitude
    curr = b2.TimedArray(tmp, dt=1. * unit_time)
    return curr

def testIt():


    # specify step current
    step_current = only_input_step_current(
        t_start=100, t_end=600, unit_time=b2.ms,
        amplitude=1.2 * b2. mamp )
    # run the LIF model

    M = simulate_WORM_neuron(input_current=step_current, simulation_time=1000 * b2.ms)

    # plot the membrane voltage

    plt.plot(M.t / b2.ms, M.v[0], label='Neuron 0')
    plt.plot(M.t / b2.ms, M.v[1], label='Neuron 1')

    #plot_tools.plot_voltage_and_current_traces(state_monitor, step_current,title="Step current")
    plt.show()

    sinusoidal_current = input_factory.get_sinusoidal_current(
        200, 800, unit_time= b2.ms,
        amplitude=2.5 * b2.mamp, frequency=5* b2.Hz, direct_current=2. * b2.namp)
    # run the LIF model
    state_monitor = simulate_1_WORM_neuron(
        input_current=sinusoidal_current, simulation_time=1000 * b2.ms)
    # plot the membrane voltage
    plot_tools.plot_voltage_and_current_traces(
        state_monitor, sinusoidal_current, title="Sinusoidal input current")
    plt.show()

if __name__ == "__main__":
    testIt()
