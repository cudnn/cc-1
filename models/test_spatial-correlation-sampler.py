import matplotlib.pyplot as plt

import  torch

img = torch.ones(3,100,100)

plt.imsave('test.png',img,cmap='plasma')
