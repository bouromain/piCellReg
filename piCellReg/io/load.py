import os.path as op
import numpy as np
import glob
import warnings


def find_file_rec(root_name: str, f_name: str, no_duplicate: bool = False):
    root_name = op.abspath(op.expanduser(root_name))
    # add trailing slash if needed
    root_name = op.join(root_name, "")
    if not op.isdir(root_name):
        raise ValueError(f"Path {root_name} shoudl be a directory")

    f = glob.glob(root_name + "**/" + f_name, recursive=True)

    # check for errors
    if not f:
        raise FileNotFoundError

    if no_duplicate:
        if len(f) > 1:
            warnings.warn(
                "Duplicate %s file found in %s only the first entry will be considered"
                % (self.root_path, file_name)
            )
        f = f[0]

    return f


### for dev
# root_path = "/Users/bouromain/Sync/tmpData/crossReg/4466"
# a = find_file_rec(root_path, "stat.npy")
# print(a)
