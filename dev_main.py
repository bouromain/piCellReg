import argparse

import matplotlib
from piCellReg.io.load import find_file_rec
from piCellReg.datatype.Session import Session
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation


p1 = "/Users/bouromain/Sync/tmpData/crossReg/4466/20201013/stat.npy"
p2 = "/Users/bouromain/Sync/tmpData/crossReg/4466/20201014/stat.npy"

s1 = Session(p1)
s2 = Session(p2)

offset = phase_cross_correlation(s1._mean_image_e, s2._mean_image_e, upsample_factor=10)


plt.figure()
plt.subplot(121)
plt.imshow(s1._mean_image_e)

plt.subplot(122)
plt.imshow(s2._mean_image_e)
