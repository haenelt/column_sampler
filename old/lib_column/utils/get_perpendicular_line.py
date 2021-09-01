def get_perpendicular_line(file_path, vtx, fac, adjm, line_length=1, line_step=0.1, nn_smooth=5,
                           ndir_smooth=5):
    """
    This function computes lines with a defined line length and step size perpendicuar to a manually
    drawn freesurfer path. Multiple paths can be saved in a signal path file. Line coordinates for
    each vertex point of each path are returned in a list [label, label point, line point].
    Inputs:
        *file_path: filename of freesurfer path file.
        *vtx: vertex coordinates.
        *fac: face array.
        *adjm: adjacency matrix.
        *line_length: length of perpendicular lines in one direction.
        *line_step: step size of perpendicular line.
        *nn_smooth: number of neighborhood iterations for smoothed surface normal. 
        *ndir_smooth: number of line neighbors in both direction for smoothed line direction.
    Outputs:
        *line_perpendicular: multidimensional list containing line coordinates for all paths.
        
    created by Daniel Haenelt
    Date created: 05-03-2020         
    Last modified: 05-10-2020
    """
    import numpy as np
    from numpy.linalg import norm
    from lib_column.io.path2label import path2label
    from gbb.normal.get_normal import get_normal
    from gbb.neighbor.nn_2d import nn_2d

    # line equation
    y = lambda x, a, b : np.array([a + x * ( a - b ) / norm(a-b) for x in x])

    # get path indices
    label = path2label(file_path)

    # get surface normals
    normal = get_normal(vtx, fac)
    
    # initialize line coordinate parameters
    x = np.arange(-line_length,line_length+line_step,line_step)
    
    # initialize list
    line_perpendicular = []

    for i in range(np.max(label["number"])):
        
        # get current line
        line_temp = []
        line = label['ind'][label['number'] == i+1]

        # get perpendicular line for each line vertex
        #for j in range(ndir_smooth,len(line)-ndir_smooth):
        for j in range(2,len(line)-2):
            
            # get local neighborhood
            nn = nn_2d(line[j], adjm, nn_smooth)
            
            # For a stable normal line computation, neighbor points in ndir_smooth distance are 
            # selected. This prevents the use of points at both line endings. For those cases, the 
            # neighbor distance is shortened.
            if j < ndir_smooth:
                p01 = np.mean(vtx[line[j+1:j+ndir_smooth],:], axis=0) - vtx[line[j],:]
                p02 = np.mean(vtx[line[0:j-1],:], axis=0) - vtx[line[j],:]
            elif j > len(line) - ndir_smooth:
                p01 = np.mean(vtx[line[j+1:len(line)],:], axis=0) - vtx[line[j],:]
                p02 = np.mean(vtx[line[j-ndir_smooth:j-1],:], axis=0) - vtx[line[j],:]
            else:
                p01 = np.mean(vtx[line[j+1:j+ndir_smooth],:], axis=0) - vtx[line[j],:]
                p02 = np.mean(vtx[line[j-ndir_smooth:j-1],:], axis=0) - vtx[line[j],:]

            # get smoothed surface normal
            p1 = np.mean(normal[nn], axis=0)
            
            # get mean vector normal to line and surface normal direction
            p21 = np.cross(p01,p1)
            p22 = np.cross(p1,p02) # flip for consistent line direction
            p2 = (p21 + p22)  / 2

            # add current line vertex point
            p0 = vtx[line[j],:]
            p2 += vtx[line[j],:]

            # get coordinates of perpendicular line
            yy = y(x,p0,p2)
            
            # append to list
            line_temp.append(yy)
        
        # get whole line into final list
        line_perpendicular.append(line_temp)
    
    return line_perpendicular
