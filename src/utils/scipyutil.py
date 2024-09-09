import scipy
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator

from .arrayutil import cache_array

@cache_array
def compute_delaunay(coord):
    return Delaunay(coord)

def compute_interpolator(coord, value, method='linear'):
    return {
        'linear': LinearNDInterpolator,
        'nearest': NearestNDInterpolator,
        'cubic': CloughTocher2DInterpolator,
    }[method](compute_delaunay(coord), value)

def interpolate(coord, value, newcoord, method='linear'):
    return compute_interpolator(coord, value, method=method)(newcoord)