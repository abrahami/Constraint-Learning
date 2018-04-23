import numpy as np

__author__ = 'abrahami'

def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)

np_version = _parse_version(np.__version__)

if np_version < (1, 6, 2):
    # Allow bincount to accept empty arrays
    # https://github.com/numpy/numpy/commit/40f0844846a9d7665616b142407a3d74cb65a040
    def bincount(x, weights=None, minlength=None):
        if len(x) > 0:
            return np.bincount(x, weights, minlength)
        else:
            if minlength is None:
                minlength = 0
            minlength = np.asscalar(np.asarray(minlength, dtype=np.intp))
            return np.zeros(minlength, dtype=np.intp)

else:
    from numpy import bincount


try:
    from scipy.special import expit     # SciPy >= 0.10
    with np.errstate(invalid='ignore', over='ignore'):
        if np.isnan(expit(1000)):       # SciPy < 0.14
            raise ImportError("no stable expit in scipy.special")
except ImportError:
    def expit(x, out=None):
        """Logistic sigmoid function, ``1 / (1 + exp(-x))``.

        See sklearn.utils.extmath.log_logistic for the log of this function.
        """
        if out is None:
            out = np.empty(np.atleast_1d(x).shape, dtype=np.float64)
        out[:] = x

        # 1 / (1 + exp(-x)) = (1 + tanh(x / 2)) / 2
        # This way of computing the logistic is both fast and stable.
        out *= .5
        np.tanh(out, out)
        out += 1
        out *= .5

        return out.reshape(np.shape(x))