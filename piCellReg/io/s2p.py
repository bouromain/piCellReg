import numpy as np
import os
import glob

p = "/home/bouromain/Sync/tmpData/crossReg/4453"


def loads2p(root_path: str, recursive=True):
    """
    load the data from suite2p output npy files
    """
    assert root_path is not None, "You should provide a root path to load data from"

    # search data from this folder
    all_stat = glob.glob(root_path + "/**/*" + "stat.npy", recursive=recursive)
    all_ops = glob.glob(root_path + "/**/*" + "ops.npy", recursive=recursive)
    # here potentially check for missing ops or stat

    # data = np.load(os.path.join(fpath), allow_pickle=True)
    # then load stat

    # load ops but only keep meanImgE Lyc Lxc refImg Ly Lx meanImg
    # add path

    # output an align session with only stat and aformentionned stuff per session
