from pathlib import Path
import os.path as op
from piCellReg.datatype.Aln import Aln
from piCellReg.datatype.Session import Session
from piCellReg.datatype.SessionPair import SessionPair
from piCellReg.registration.utils import shift_image


user_path = str(Path.home())

p1 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201013/stat.npy")
p2 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201014/stat.npy")

s1 = Session(p1)
s2 = Session(p2)
s_p = SessionPair(s1, s2)
dists = s_p.distcenters()
ov = s_p.overlaps()
C = s_p.correlations()


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(s1._mean_image_e, cmap="Reds")
s2_s = shift_image(s2._mean_image_e, -s_p._relative_offsets)

plt.imshow(s2_s, alpha=0.5, cmap="Greens")

plt.subplot(1, 2, 2)
plt.imshow(s1.get_projection(), cmap="Reds", interpolation="nearest")
plt.imshow(
    s2.get_projection(
        x_shift=s_p._relative_offsets[1],
        y_shift=s_p._relative_offsets[0],
        theta=s_p._rotation,
    ),
    alpha=0.5,
    cmap="Greens",
    interpolation="nearest",
)
plt.axis("off")


plt.show()
