#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
from piCellReg.io.load import find_file_rec
from piCellReg.datatype.Session import Session
from piCellReg.datatype.Aln import Aln
import matplotlib.pyplot as plt


def scan_stat_files(root_path, verbose=True):
    # list_stat = find_file_rec(root_path, "stat.npy")
    # if verbose:
    #     print(f"{len(list_stat)} sessions found, initialing ...")

    # list_sessions = [Session(p) for p in list_stat]
    # print(list_stat[1])
    all_session = Aln(root_path)
    if all_session._registration.isempty():
        all_session.register()
        print(
            f"Best reference session is session {all_session._registration.best_ref()}"
        )
    # Now we now the offsets calculate the correlation and
    # distance between all the sessions and ref session


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", help="Source folder")
    parser.add_argument("-v", "--verbose", help="make the script chatty", default=True)

    args = vars(parser.parse_args())
    scan_stat_files(**args)
