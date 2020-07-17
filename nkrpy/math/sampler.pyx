"""Sampler."""
# flake8: noqa

# internal modules

# external modules
import numpy as np

# relative modules

# global attributes
__all__ = ('gaussian_sample', 'sampler', 'samplers')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
samplers = ('gaussian', 'uniform')

def gaussian_sample(lower_bound, upper_bound, size: int=100, scale=None):
    """Sample from a gaussian given limits."""
    if lower_bound == upper_bound:
        scale = 0
    loc = (lower_bound + upper_bound) / 2.
    if scale is None:
        scale = (upper_bound - lower_bound) / 2.
    results = []
    while len(results) < size:
        samples = np.random.normal(loc=loc, scale=scale,
                                   size=size - len(results))
        results += [sample for sample in samples
                    if lower_bound <= sample <= upper_bound]
    return results


def _sample(*args, sample, **kwargs):
    if sample in ('gaussian', 'normal'):
        ret = gaussian_sample(*args, **kwargs)
    elif sample == 'uniform':
        ret = np.random.uniform(*args, **kwargs)
    return ret


def sampler(*args, sample: str='gaussian', resample=False,
           lim=None, logic=None, resample_n=100, **kwargs):
    """Sample a given distribution.

    Parameters
    ----------
    sampler: str
        must be in the list of common samples
    resample: bool
        if resampling is allowed
    lim: float | None
        An upper or lower limit, if outside then resamples
    logic: str [< | >]
        Determines whether lim is upper or lower lim
    resample_n: int
        Number of resamples to try, worse case
    """
    sample = sample.lower()
    ret = np.array(_sample(*args, sample=sample, **kwargs))
    dshape = ret.shape
    fine = np.zeros(1)
    i = 0
    while resample and (i < resample_n):
        gatherl = np.where(ret < lim)[0]
        gatherg = np.where(ret > lim)[0]
        if (('<' in logic) and (len(gatherl) < dshape[0])) or \
           (('>' in logic) and (len(gatherg) < dshape[0])):
            resample = True
        else:
            resample = False
        if ('>' in logic):
            gather = gatherg
        else:
            gather = gatherl
        ret = ret[gather]
        if resample:
            if gather.shape[0] < dshape[0]:
                new = np.array(_sample(*args, sample=sample, **kwargs))
                ret = np.concatenate([ret, new])
            else:
                resample = False
        i += 1
    return ret[:dshape[0]]

# end of code

# end of file
