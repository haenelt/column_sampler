import numpy as np
from column_sampler.mesh import Mesh

# input
vtx = None
fac = None
arr_ref = None
ind = None

# parameters
line_length = 2
line_step = 0.1
nn_smooth = 0
ndir_smooth = 5

# x-coordinates
x = np.arange(-line_length,
              line_length+line_step,
              line_step)

# get adjavency matrix
mesh = Mesh(vtx, fac)
adjm = mesh.adjm

# get perpendicular lines
line = get_perpendicular_line(file_path,
                              mesh["vtx"],
                              mesh["fac"],
                              adjm,
                              line_length,
                              line_step,
                              nn_smooth,
                              ndir_smooth)

for i in range(len(line)):
    for j in range(len(line[i])):

        # sample reference data
        data = sample_line(line[i][j], file_ref, arr_ref)
        shift = get_line_shift(data)  # get shift
        print("Shift for point "+str(j)+": "+str(shift))
        line[i][j] = apply_line_shift(line[i][j],  # apply shift
                                      shift,
                                      line_length,
                                      line_step)
        
        # sample data to shifted line for sanity check
        data = sample_line(line[i][j],  # sample data for sanity check
                           file_ref,
                           arr_ref)

# plot sampled data of shifted line for sanity check
# save line coordinates
# save line coordinates as mesh







def get_perpendicular_line(file_path, vtx, fac, adjm, line_length=1,
                           line_step=0.1, nn_smooth=5,
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
    y = lambda x, a, b: np.array([a + x * (a - b) / norm(a - b) for x in x])

    # get path indices
    label = path2label(file_path)

    # get surface normals
    normal = get_normal(vtx, fac)

    # initialize line coordinate parameters
    x = np.arange(-line_length, line_length + line_step, line_step)

    # initialize list
    line_perpendicular = []

    for i in range(np.max(label["number"])):

        # get current line
        line_temp = []
        line = label['ind'][label['number'] == i + 1]

        # get perpendicular line for each line vertex
        # for j in range(ndir_smooth,len(line)-ndir_smooth):
        for j in range(2, len(line) - 2):

            # get local neighborhood
            nn = nn_2d(line[j], adjm, nn_smooth)

            # For a stable normal line computation, neighbor points in ndir_smooth distance are
            # selected. This prevents the use of points at both line endings. For those cases, the
            # neighbor distance is shortened.
            if j < ndir_smooth:
                p01 = np.mean(vtx[line[j + 1:j + ndir_smooth], :],
                              axis=0) - vtx[line[j], :]
                p02 = np.mean(vtx[line[0:j - 1], :], axis=0) - vtx[line[j], :]
            elif j > len(line) - ndir_smooth:
                p01 = np.mean(vtx[line[j + 1:len(line)], :], axis=0) - vtx[
                                                                       line[j],
                                                                       :]
                p02 = np.mean(vtx[line[j - ndir_smooth:j - 1], :],
                              axis=0) - vtx[line[j], :]
            else:
                p01 = np.mean(vtx[line[j + 1:j + ndir_smooth], :],
                              axis=0) - vtx[line[j], :]
                p02 = np.mean(vtx[line[j - ndir_smooth:j - 1], :],
                              axis=0) - vtx[line[j], :]

            # get smoothed surface normal
            p1 = np.mean(normal[nn], axis=0)

            # get mean vector normal to line and surface normal direction
            p21 = np.cross(p01, p1)
            p22 = np.cross(p1, p02)  # flip for consistent line direction
            p2 = (p21 + p22) / 2

            # add current line vertex point
            p0 = vtx[line[j], :]
            p2 += vtx[line[j], :]

            # get coordinates of perpendicular line
            yy = y(x, p0, p2)

            # append to list
            line_temp.append(yy)

        # get whole line into final list
        line_perpendicular.append(line_temp)

    return line_perpendicular


def sample_line(coords, file_data, data_array):
    """
    This function samples data onto the given coordinate array using linear interpolation.
    Inputs:
        *coords: coordinate array.
        *file_data: filename of nifti volume.
        *data_array: array with data points of nifti volume.
    Outputs:
        *data_sampled: array with sampled data.

    created by Daniel Haenelt
    Date created: 05-03-2020
    Last modified: 13-10-2020
    """
    from nibabel.affines import apply_affine
    from gbb.interpolation.linear_interpolation3d import linear_interpolation3d
    from gbb.utils.vox2ras import vox2ras

    # get ras to voxel transformation
    _, ras2vox_tkr = vox2ras(file_data)

    # apply transformation to input coordinates
    coords = apply_affine(ras2vox_tkr, coords)

    # sample data
    data_sampled = linear_interpolation3d(coords[:, 0],
                                          coords[:, 1],
                                          coords[:, 2],
                                          data_array)

    return data_sampled


def get_line_shift(data):
    """
    This function looks for a peak in data sampled in 1D. If multiple peaks are found, only the
    peak closest to the line center is considered.
    Inputs:
        *data: 1D array of sampled data.
    Outputs:
        *data_shift: shift of data peak relative to line center.

    created by Daniel Haenelt
    Date created: 05-03-2020
    Last modified: 09-03-2020
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
    y = lambda x, a, b: np.array([a + x * (a - b) / norm(a - b) for x in x])

    # initialize line coordinate parameters
    x = np.arange(-line_length, line_length + line_step, line_step)

    # center coordinate
    xc = int(len(x) / 2)

    # new center coordinate
    xc = xc + shift

    # get coordinates of perpendicular line
    yy = y(x, coords[xc, :], coords[xc + 1, :])

    return yy