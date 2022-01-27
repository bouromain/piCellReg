from piCellReg.datatype import Session
from piCellReg.registration.register import register_im


class PairedSession:
    def __init__(self, *, s0: Session = None, s0_idx=0, s1: Session = None, s1_idx=1):
        self.sess0 = s0
        self.sess1 = s1
        self.sess0_index = s0_idx
        self.sess1_index = s1_idx
        self.offset = []
        self.rotation = 0

    def register(self, method="rigid"):
        if method == "rigid":
            offset = register_im(self.sess0._mean_image_e, self.sess1._mean_image_e)
        elif method == "rigid_rotation":
            ...

        else:
            raise NotImplementedError(f"Method {method} not implemented")

