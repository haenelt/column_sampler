def path2label(path_in, label_out=None):
    """
    This function function separates paths saved in a path file into different
    labels. Optionally, a label file is written which contains label indices
    and annotations of all paths.
    Inputs:
        *path_in: filename of input path file.
        *label_out: filename of output label file (if not None).
    Outputs:
        *label: dict containing label indices 'ind' and label number 'number'.
        
    created by Daniel Haenelt
    Date created: 04-03-2020          
    Last modified: 04-03-2020
    """
    import numpy as np

    # read path file
    fileID = open(path_in, "r")

    # skip path header and get number of points for each path
    data = []
    num_points = []
    access = 0
    line = fileID.readline()
    while line:
        line = fileID.readline()
        if "ENDPATH" in line:
            access = 0
        elif access:
            data = np.append(data, line.split()[-1])
        elif "NUMPOINTS" in line:
            num_points = np.append(num_points, line.split()[-1])
            access = 1
    
    fileID.close()

    # convert to integer
    data = data.astype(int)
    num_points = num_points.astype(int)

    # label array
    label = {}
    label['ind'] = []
    label['number'] = []

    # write label file
    if label_out:
        fileID = open(label_out, "w")    
        fileID.write("#!ascii label , from subject daniel vox2ras=TkReg coords=white\n")
        fileID.write(str(len(data))+"\n")
    
    num_counter = 0    
    c = 0
    for i in range(len(data)):
        if label_out:
            fileID.write(str(data[i])+" 0.000 0.000 0.000 "+str(num_counter+1)+"\n")
        label['ind'] = np.append(label['ind'], data[i].astype(int))
        label['number'] = np.append(label['number'], num_counter+1)
        
        c += 1
        if c == num_points[num_counter]:
            c = 0
            num_counter += 1

    if label_out:
        fileID.close()
    
    # convert to integer
    label['ind'] =label['ind'].astype(int)
    label['number'] = label['number'].astype(int)
    
    return label