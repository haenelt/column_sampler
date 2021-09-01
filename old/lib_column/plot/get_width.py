def get_width(x, y, invert_peak=False):
    """
    From columnar profiles, the mean width across layers is computed.
    Inputs:
        *x: array containing x-coordinates.
        *y: array containing y-coordinates per layer (layer x coordinates).
        *invert_peak: invert y-coordinates.
    Outputs:
        *hwhm_left: left column size from peak center.
        *hwhm_right: right column size from peak center.

    created by Daniel Haenelt
    Date created: 01-07-2020
    Last modified: 01-07-2020
    """
    import numpy as np

    # get some parameters
    xc = np.where(np.abs(x)<1e-6)[0][0]
    n_layer = np.shape(y)[0]
    
    if invert_peak:
        y = -1 * y
    
    hwhm_left = []
    hwhm_right = []
    for i in range(n_layer):
        
        # get column peak value
        y_max = y[i,xc]

        # get right minimum
        y_temp = y[i,xc:].copy()
        yr_min = np.nanmin(y_temp)
        yr_threshold = np.mean([y_max,yr_min])
        
        # get left minimum
        y_temp = y[i,:xc].copy() 
        yl_min = np.nanmin(y_temp)
        yl_threshold = np.mean([y_max, yl_min])
    
        # column width (fwhm)
        n = xc
        while True:
            if n > len(y[i,:]):
                yr_hwhm = 0
                break
            elif y[i,n] <= yr_threshold:
                yr_hwhm = n
                break
            
            n += 1
    
        # column width (fwhm)
        n = xc
        while True:
            if n < 0:
                yl_hwhm = 0
                break
            elif y[i,n] <= yl_threshold:
                yl_hwhm = n
                break
        
            n -= 1
        
        # get hwhm if exists for both sides
        if yl_hwhm and yr_hwhm:
            hwhm_left.append(yl_hwhm)
            hwhm_right.append(yr_hwhm)
    
    # mean hwhm as displacement from center
    hwhm_left = np.mean(hwhm_left) / len(x) * ( x[-1] - x[0]) - x[-1]
    hwhm_right = np.mean(hwhm_right) / len(x) * ( x[-1] - x[0] ) - x[-1]
    
    return hwhm_left, hwhm_right