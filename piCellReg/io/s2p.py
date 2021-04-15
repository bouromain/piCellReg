import numpy as np
import os


def default_aln():
    return {
        "idx_session": [],
        "stat": [],
        "meanImg": [],
        "meanImgE": [],
        "xoffset": [],
        "yoffset": [],
        "isref": False,
        "stat_path": [],
        "ops_path": [],
    }


def loads2p(root_path: str, recursive=True):
    """
    load the data from suite2p output npy files
    """
    assert root_path is not None, "You should provide a root path to load data from"

    # Verify that we have a stat file and an ops file too
    root_path = os.path.expanduser(root_path)

    assert os.path.exists(
        os.path.join(root_path, "stat.npy")
    ), "stat.npy file not found"
    assert os.path.exists(os.path.join(root_path, "ops.npy")), "ops.npy file not found"

    # generate and fill aln
    aln = aln_load(root_path)

    return aln


def aln_load(fpath):
    """
    load data for an aln
    """
    fpath = os.path.expanduser(fpath)

    aln = default_aln()
    # populate aln with stats
    aln["stat"] = np.load(os.path.join(fpath, "stat.npy"), allow_pickle=True)
    aln["stat_path"] = fpath

    # now fill it with the ops info
    tmp_ops = np.load(os.path.join(fpath, "ops.npy"), allow_pickle=True)
    tmp_ops = tmp_ops.item()
    aln["meanImg"] = tmp_ops["meanImg"]
    aln["meanImgE"] = tmp_ops["meanImgE"]
    aln["ops_path"] = fpath

    return aln


p = [
    "~/Sync/tmpData/crossReg/4466/20201010",
    "~/Sync/tmpData/crossReg/4466/20201011",
    "~/Sync/tmpData/crossReg/4466/20201013",
]
aln = [[] for _ in p]
for it, val in enumerate(p):
    aln[it] = loads2p(val)
    aln[it]["idx_session"] = it
