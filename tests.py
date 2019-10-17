import numpy as np

a = np.array([1,2,3,4,1,2,3,1,1,4])

h,b = np.histogram(a,bins=100,range=[0,100])
import torch

print(h)

print('ok')
