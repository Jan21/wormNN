import random
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

def sigmoid(x, sigma, mu):
    return 1 / (1 + math.exp(-sigma * (x - mu)))


class Neuron:
    def __init__(self, rest, length):
        self.v = np.ones(length) * rest
        self.exCons = {}
        self.inhCons = {}
        self.gjCons = {}
        self.input = -1


def createNeurons(num, rest, length):
    neurons = []
    for i in range(num):
        neurons.append(Neuron(rest, length))
    return neurons


def createSynapses(neurons, synapses):
    for syn in synapses['inp']:
        neurons[syn[1]].input = syn[0]
    for syn in synapses['inh']:
        neurons[syn[1]].inhCons[syn[0]] = syn[2]
    for syn in synapses['ex']:
        neurons[syn[1]].exCons[syn[0]] = syn[2]
    for syn in synapses['gj']:
        neurons[syn[1]].gjCons[syn[0]] = syn[2]


def testParamSetting(testingParams, topology, inp):
    G_leak= testingParams['G_leak']
    V_leak= testingParams['V_leak']
    C_m= testingParams['C_m']
    delT= testingParams['delT']
    E_rev_in= testingParams['E_rev_in']
    E_rev_ex= testingParams['E_rev_ex']
    sigma= testingParams['sigma']
    mu= testingParams['mu']
    rest= testingParams['rest']
    def eulerStepForNeuron(t, neuron, neurons,inputs):
        summedInh = 0
        summedEx = 0
        summedGJ = 0
        summedGJWithPre = 0
        if neuron.input != -1:
            neuron.v[t + 1] = (((C_m / delT) *
                                neuron.v[
                                    t] + G_leak * V_leak + summedInh * E_rev_in + summedEx * E_rev_ex + summedGJWithPre) / \
                               (C_m / delT + G_leak + summedInh + summedEx + summedGJ))+inputs[neuron.input, t]
            return
        for inh,w in neuron.inhCons.items():
            summedInh += sigmoid(neurons[inh].v[t], sigma, mu) * w
        for ex,w in neuron.exCons.items():
            summedEx += sigmoid(neurons[ex].v[t], sigma, mu) * w
        for gj,w in neuron.gjCons.items():
            summedGJ += w
            summedGJWithPre += w * neurons[gj].v[t]
        neuron.v[t + 1] = (((C_m / delT) *
        neuron.v[t] + G_leak * V_leak + summedInh * E_rev_in + summedEx * E_rev_ex + summedGJWithPre) / \
                          (C_m / delT + G_leak + summedInh + summedEx + summedGJ))
        return

    inp_len = len(inp[0, :])
    neurons = createNeurons(topology['num'], rest, inp_len)
    createSynapses(neurons, topology['syn'])

    for t in range(inp_len-1):
        for i, neuron in enumerate(neurons):
            eulerStepForNeuron(t, neuron, neurons, inp)
    return neurons



#default params



simulationRanges={
    'G_leak' : (0.1,6 ),
    'V_leak' :  (-0.075,-0.065 ),
    'C_m' : (0.01, 1),
    'delT' : (.001, .01),
    'E_rev_in' :  (1, 1),
    'E_rev_ex' :  (1, 1),
    'sigma' :  (1, 1),
    'mu' : (1, 1),
    'rest' : (1, 1)
}

defaultParams={
    'G_leak' :  1,
    'V_leak' : -0.07,
    'C_m' : 0.05,
    'delT' : 0.01,
    'E_rev_in' : -0.09,
    'E_rev_ex' : 0,
    'sigma' : 250,
    'mu' : -0.02,
    'rest' : -0.07
}


topology = {
    'num': 3,
    'syn': {
        'inp': [(0,0)],  # input -> neuron
        'inh': [],  # input neuron , output neuron , weigth (1, 2,1)
        'ex': [(0, 1,1000)],  # input neuron , output neuron , weigth
        'gj': []  # input neuron , output neuron , weigth
    }
}

length = 400
inp = np.ones([1,length])*0
inp[0,100:200]=.01

#inp[0,100:200]=np.linspace(0,0.1,100)
#inp[0,300:400]=np.linspace(0.1,0,100)


def testRanges3D(params,points_per_axis):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    paramRanges={
        'px':[],
        'py':[],
        'pz':[],
    }
    resultsOfSim = np.zeros([4,points_per_axis,points_per_axis,points_per_axis])

    for i,pR in enumerate(paramRanges.keys()):
        lower_bound = simulationRanges[params[i]][0]
        upper_bound = simulationRanges[params[i]][1]
        paramRanges[pR] = np.random.uniform(lower_bound,upper_bound,points_per_axis)
    for i, p0 in enumerate(paramRanges['px']) :
        for j, p1 in enumerate(paramRanges['py']):
            for k, p2 in enumerate(paramRanges['pz']):
                index=i*j*k
                defaultParams[params[0]]=p0
                defaultParams[params[1]]=p1
                defaultParams[params[2]]=p2
                neurons = testParamSetting(defaultParams, topology, inp)
                resultsOfSim[3][i][j][k]= np.max(neurons[1].v)
                resultsOfSim[0][i][j][k]= p0
                resultsOfSim[1][i][j][k]= p1
                resultsOfSim[2][i][j][k]= p2


    minim=np.min(resultsOfSim[3,:])
    maxim=np.max(resultsOfSim[3,:])

    normalize = matplotlib.colors.Normalize(vmin=minim, vmax=maxim)
    for i,v1 in enumerate(resultsOfSim[0,:,0,0]):
        for j, v2 in enumerate(resultsOfSim[0, 0, :, 0]):
            for k, v3 in enumerate(resultsOfSim[0, 0, 0, :]):
                ax.scatter(resultsOfSim[0][i][j][k], resultsOfSim[1][i][j][k], resultsOfSim[2][i][j][k], c=resultsOfSim[3][i][j][k],cmap='bone',norm=normalize
                           )

    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    ax.set_zlabel(params[2])
    plt.show()
    print resultsOfSim

    return resultsOfSim

#results=testRanges3D(['G_leak','V_leak','C_m'],5)

def testRanges2D(params,points_per_axis):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    paramRanges={
        'px':[],
        'py':[],
    }
    resultsOfSim = np.zeros([3,points_per_axis,points_per_axis])

    for i,pR in enumerate(paramRanges.keys()):
        lower_bound = simulationRanges[params[i]][0]
        upper_bound = simulationRanges[params[i]][1]
        paramRanges[pR] = np.linspace(lower_bound,upper_bound,points_per_axis)
    for i, p0 in enumerate(paramRanges['px']) :
        for j, p1 in enumerate(paramRanges['py']):

                defaultParams[params[0]]=p0
                defaultParams[params[1]]=p1
                neurons = testParamSetting(defaultParams, topology, inp)
                resultsOfSim[2][i][j]= np.max(neurons[1].v)
                resultsOfSim[0][i][j]= p0
                resultsOfSim[1][i][j]= p1


    minim=np.min(resultsOfSim[2,:])
    maxim=np.max(resultsOfSim[2,:])
    print 'minim:', minim, " maxim:", maxim
    normalize = matplotlib.colors.Normalize(vmin=minim, vmax=maxim)
    for i,v1 in enumerate(resultsOfSim[0,:,0]):
        for j, v2 in enumerate(resultsOfSim[0, 0, : ]):
                ax.pcolor(resultsOfSim[0][i][j], resultsOfSim[1][i][j], resultsOfSim[2][i][j],cmap='cool',norm=normalize
                           )

    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    plt.show()

    return resultsOfSim

#results=testRanges2D(['G_leak','C_m'],30)


# statistics functions

def get_rise_time(neurons):
    max_v=np.max(neurons)
    start_v=neurons[0]
    above_start=np.argmax(neurons > (start_v + abs(start_v - max_v)*0.1))
    above_max=np.argmax(neurons > (start_v + abs(start_v - max_v)*0.9))
    return above_max-above_start




def testRanges2DColorPlot(params,points_per_axis,statisticsFn,num_of_neuron,multiplication_Constant):

    plt.subplot(1, 1, 1)

    paramRanges={
        'px':[],
        'py':[],
    }
    resultsOfSim = np.zeros([points_per_axis,points_per_axis])

    for i,pR in enumerate(paramRanges.keys()):
        lower_bound = simulationRanges[params[i]][0]
        upper_bound = simulationRanges[params[i]][1]
        paramRanges[pR] = np.linspace(lower_bound,upper_bound,points_per_axis)
    for i, p0 in enumerate(paramRanges['px']) :
        for j, p1 in enumerate(paramRanges['py']):

                defaultParams[params[0]]=p0
                defaultParams[params[1]]=p1
                neurons = testParamSetting(defaultParams, topology, inp)
                resultsOfSim[i][j]= statisticsFn(neurons[num_of_neuron].v) * multiplication_Constant


    minim=np.min(resultsOfSim[:])
    maxim=np.max(resultsOfSim[:])
    print 'minim:', minim, " maxim:", maxim

    plt.pcolor(paramRanges['px'], paramRanges['py'], resultsOfSim, cmap='RdBu', vmin=minim, vmax=maxim)
    plt.title('pcolor')
    # set the limits of the plot to the limits of the data
    plt.axis([paramRanges['px'].min(), paramRanges['px'].max(), paramRanges['py'].min(), paramRanges['py'].max()])
    plt.colorbar()


    plt.show()

    return resultsOfSim

results=testRanges2DColorPlot(['G_leak','C_m'],30,get_rise_time,1,0.01)