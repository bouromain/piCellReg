from piCellReg.io.load import find_file_rec
from piCellReg.datatype.Session import Session
from piCellReg.registration.utils import shift_image
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
import numpy as np

p1 = "/Users/bouromain/Sync/tmpData/crossReg/4466/20201013/stat.npy"
p2 = "/Users/bouromain/Sync/tmpData/crossReg/4466/20201014/stat.npy"

s1 = Session(p1)
s2 = Session(p2)

offset = phase_cross_correlation(
    s1._mean_image_e, s2._mean_image_e, upsample_factor=100
)
im_s = shift_image(s2._mean_image_e, -offset[0])

plt.figure(figsize=(20, 20))
plt.subplot(221)
plt.imshow(s1._mean_image_e)

plt.subplot(222)
plt.imshow(s2._mean_image_e)

plt.subplot(223)
plt.imshow(s1._mean_image_e)
plt.imshow(s2._mean_image_e, alpha=0.5, cmap="magma")

plt.subplot(224)
plt.imshow(s1._mean_image_e)
plt.imshow(im_s, alpha=0.5, cmap="magma")

