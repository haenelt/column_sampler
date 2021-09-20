def get_line_shift(data):
    """
    This function looks for a peak in data sampled in 1D. If multiple peaks are found, only the 
    peak closest to the line center is considered.
    Inputs:
        *data: 1D array of sampled data.
    Outputs:
        *data_shift: shift of data peak relative to line center.
    """
    import numpy as np
    from scipy.signal import find_peaks

    # center coordinate
    xc = int(len(data) / 2)

    # get line peaks
    peaks, _ = find_peaks(data)

    # get peak closest to center coordinate
    if len(peaks) > 1:
        peaks_diff = np.abs(peaks - xc)
        peaks = peaks[np.where(peaks_diff == np.min(peaks_diff))[0][0]]
        
        # get distance to center coordinate
        data_shift = peaks - xc
        
    elif len(peaks) == 1:
        peaks = peaks[0]
        
        # get distance to center coordinate
        data_shift = peaks - xc
        
    else:
        data_shift = []


    return data_shift