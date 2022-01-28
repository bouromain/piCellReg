from pathlib import Path
import os.path as op
from piCellReg.datatype.Aln import Aln
from piCellReg.datatype.Session import Session
from piCellReg.datatype.SessionPair import SessionPair


user_path = str(Path.home())

p1 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201013/stat.npy")
p2 = op.join(user_path, "Sync/tmpData/crossReg/4466/20201014/stat.npy")

s1 = Session(p1)
s2 = Session(p2)

###
p_all = op.join(user_path, "Sync/tmpData/crossReg/4466/")
s_all = Aln(p_all)
