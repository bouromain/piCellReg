from pathlib import Path
import os.path as op
from itertools import combinations
from piCellReg.datatype.Session import Session, SessionList
from piCellReg.datatype.SessionPair import SessionPair
from piCellReg.registration.utils import shift_image
import bottleneck as bn

# user_path = str(Path.home())

# p1 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201013/stat.npy")
# p2 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201014/stat.npy")

# s1 = Session(p1)
# s2 = Session(p2)
# s_p = SessionPair(s1, s2)
# dists = s_p.distcenters()
# ov = s_p.overlaps()
# C = s_p.correlations()

## List of all sessions pairs
# make a list of sessions
user_path = str(Path.home())

p_all = op.join(user_path, "Sync/tmpData/crossReg/4466/")
sess_list = SessionList().load_from_s2p(p_all)

# make a list of all SessionPair with all the combinations of Session possible
L = [
    SessionPair(s0, s1, id_s0=i0, id_s1=i1)
    for ((i0, s0), (i1, s1)) in combinations(enumerate(sess_list), 2)
]

all_distances = [l.correlations() for l in L]

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
