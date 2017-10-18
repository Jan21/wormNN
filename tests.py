import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x,sigma,mu):
  return 1 / (1 + math.exp(-sigma*(x-mu)))


v=np.linspace(-0.07,0,100)
o=np.zeros(100)
sigma = 250
mu = -0.02
for i in range(len(o)):
    o[i] = sigmoid(v[i],sigma,mu)

plt.plot(v,o)
plt.show()