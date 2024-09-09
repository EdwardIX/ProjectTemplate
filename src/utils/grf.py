from gstools import SRF, Gaussian
from .arrayutil import grid

class GaussianRF(object):

    def __init__(self, dim, var=2, len_scale=3):
        self.model = Gaussian(dim=dim, var=var, len_scale=len_scale)

    def sample(self, shape, N):
        x = grid(shape)
        sample = []
        for i in range(N):
            srf = SRF(self.model)
            sample.append(srf(x))
        
        return np.stack(sample)