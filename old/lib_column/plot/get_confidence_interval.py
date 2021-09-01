def get_confidence_interval(se, n, c=0.95):
    """
    Computation of confidence interval.
    Inputs:
        *c: confidence threshold.
        *n: length of data array.
        *se: standard error of the mean.

    created by Daniel Haenelt
    Date created: 09-03-2020
    Last modified: 09-03-2020
    """
    from scipy.stats import t
    
    ci = se * t.ppf((1 + c) / 2, n - 1)
    
    return ci