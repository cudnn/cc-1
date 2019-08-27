import numpy as np

from scipy.misc import imread
import custom_transforms

trans = custom_transforms.Compose([
custom_transforms.ArrayToTensor(),
    custom_transforms.Normalize()

])

tgt_depth = np.load('/home/roit/datasets/MC_256512/2019_08_26_10_05_26/depths/0000057.npy')
tgt_depth = np.expand_dims(tgt_depth,axis=2)


intr = np.array([241.674463,0.,204.168010,0.,246.284868,59.000832,0.,0.,1.]).reshape(3,3)
out = trans([tgt_depth],intr)


print('ok')

