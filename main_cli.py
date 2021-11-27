#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import argparse
from piCellReg.io.load import find_file_rec
from piCellReg.datatype.Session import Session
import matplotlib.pyplot as plt
import time


def scan_stat_files(root_path, verbose=True):
    list_stat = find_file_rec(root_path, "stat.npy")
    if verbose:
        print(f"{len(list_stat)} sessions found, initialing ...")

    list_sessions = [Session(p) for p in list_stat]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", help="Source folder")
    parser.add_argument("-v", "--verbose", help="make the script chatty", default=True)

    args = vars(parser.parse_args())
    scan_stat_files(**args)
