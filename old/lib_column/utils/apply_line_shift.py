def apply_line_shift(coords, shift, line_length=2, line_step=0.1):
    """
    This function applies a shift to a line of coordinates. 
    Inputs:
        *coords: coordinate array.
        *shift: amount of shift in units of array steps.
        *line_length: length of perpendicular lines in one direction.
        *line_step: step size of perpendicular line.
    Outputs:
        *yy: shifted coordinate array.
        
    created by Daniel Haenelt
    Date created: 05-03-2020           
    Last modified: 05-03-2020
    """
    import numpy as np
    from numpy.linalg import norm

    # line equation
    y = lambda x, a, b : np.array([a + x * ( a - b ) / norm(a-b) for x in x])
    
    # initialize line coordinate parameters
    x = np.arange(-line_length,line_length+line_step,line_step)
            
    # center coordinate
    xc = int(len(x) / 2)
    
    # new center coordinate
    xc = xc + shift
    
    # get coordinates of perpendicular line
    yy = y(x,coords[xc,:],coords[xc+1,:])
    
    return yy