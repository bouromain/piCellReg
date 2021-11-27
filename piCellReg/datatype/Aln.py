import numpy as np
from registration.register import register_im
from dataclasses import dataclass


@dataclass
class Aln:
    """
    Alignment object
    """

    # general infos
    _sessions = []

    # registration
    _reference_session = None
    _register_matrix = None  # quality of registration between all the pairs
    _x_off_matrix = None  # x offset between all sessions pairs
    _y_off_matrix = None  # y offset between all sessions pairs
    _rot_matix = None  # rotation  between all sessions pairs
    _register_method = None  # rigid, rigid_rotation, non_rigid

    # probabilities

    @property
    def n_session(self):
        return len(self.sessions)

    def __len__(self):
        return self.n_session

    def register():
        ...
        # register all the pairs of sessions possible
