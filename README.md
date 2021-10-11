# Column sampler

<p align="center">
  <img src="https://github.com/haenelt/column_filter/blob/main/img/odc.gif?raw=true" width=75% height=75% alt="Illustration of GBB"/>
</p>

Cortical columns are often thought as the fundamental building blocks for cortical information processing. Recent advances in ultrahigh field functional magnetic resonance imaging (fMRI) enabled the mapping of cortical columns in living humans. However, fMRI still suffers from various noise sources which degrades the specificity of the signal and challenges the visualization of fine-grained cortical structures.

This package aims to uncover underlying repetitive columnar structures using a wavelet approach. A filter bank consisting of complex-valued Gabor wavelets with different spatial frequencies and orientations is applied to fMRI contrasts sampled on a surface mesh. This is similar to [[1]](#1). However, the current implementation works completely on the curved irregular triangle mesh. This has the advantage that no further data manipulation has to be done which would lead to spatial distortions, e.g. surface flattening. For each vertex in a region of interest, the filtered contrast is computed by finding the wavelet with the highest correlation to the underlying structure. The filtered contrast from the complete region of interest can then be used to visually inspect for the existence of any repetitive structures. The spatial frequency distribution allows the analysis of column width. This method might therefore be used to explore unknown cortical areas for fine-grained repetitive structures.

The figure above shows human ocular dominance columns (contrast: left eye response > right eye response, yellow > blue) in the primary visual cortex (V1) acquired at 7 T. The data is illustrated for one hemisphere on an inflated surface mesh only covering the occipital lobe. Please note that while the data is visualized on an inflated mesh, the filtering procedure was performed on the original curved mesh. The repetitive structure of ocular dominance columns can be clearly seen after filtering. The column width (half the wavelength expectation value) is 1.5 mm in line with the known size of human ocular dominance columns (see [[2]](#2)).

Another example can be found in the *img* folder which shows thin stripes in the secondary visual cortex (V2). These columnar stripes were mapped by exploiting their color sensitivity. The column width is 2.77 mm which again is in line with known sizes of human stripes (see) [[2]](#2)).

## Installation
I recommend to use `Miniconda` to create a new python environment with `python=3.8`. To install the package, clone the repository and run the following line from the directory in which the repository was cloned

```shell
python setup.py install
```

After the installation you should be able to import the package with `import column_sampler`.

## Some snippets

###  Filter data
```python
from column_filter.io import load_roi, load_mesh, load_mmap, load_overlay
from column_filter.filt import Filter

surf_in = ""  # input surface mesh
label_in = ""  # input region of interest
dist_in = ""  # input distance matrix
arr_in = ""  # input overlay
file_out = ""  # output overlay

label = load_roi(label_in)
surf = load_mesh(surf_in)
dist = load_mmap(dist_in)
arr = load_overlay(arr_in)['arr']

filter_bank = Filter(surf['vtx'], surf['fac'], label, dist)
_ = filter_bank.fit(arr, file_out=file_out)
```

### Visualize Gabor wavelet
```python
import numpy as np
from column_filter.io import load_roi, load_mesh, save_overlay, load_mmap
from column_filter.filt import Filter

surf_in = ""  # input surface mesh
label_in = ""  # input region of interest
dist_in = ""  # input distance matrix
file_out = ""  # output overlay

sigma = 1.0  # standard deviation of gaussian hull
sf = 2.0  # spatial frequency
i = 4219  # vertex index in region of interest

label = load_roi(label_in)
surf = load_mesh(surf_in)
dist = load_mmap(dist_in)

filter_bank = Filter(surf['vtx'], surf['fac'], label, dist)
coords = filter_bank.get_coordinates(i)
res = filter_bank.generate_wavelet(coords[0], coords[1], sigma, sf, 0, 0.1)
save_overlay(file_out, np.real(res))
```

```python
# Mesh
import os
from nibabel.freesurfer.io import read_geometry
from column_sampler.io import load_coords, save_coords, coords_to_mesh
from column_sampler.mesh import PlanarMesh, CurvedMesh
from column_sampler.layer import Layer

surf_in = "/data/pt_01880/zzz/lh.layer_5"
ind = [118288, 124379, 133547, 19521, 133796]

line_out = ""
plane_out = ""
curv_out = ""
coords_out = ""

# load mesh
v, f = read_geometry(surf_in)

# save line
plane = PlanarMesh(v, f, ind)
plane.save_line(line_out)  # save line as overlay
coords_to_mesh(plane_out, plane.line_coordinates)  # save plane as mesh

curv = CurvedMesh(v, f, ind)
coords = curv.project_coordinates_sequence()
coords_to_mesh(curv_out, coords)
save_coords(coords_out, coords)


# layers
file_white = "/data/pt_01880/zzz/lh.layer_0"
file_middle = "/data/pt_01880/zzz/lh.layer_5"
file_pial = "/data/pt_01880/zzz/lh.layer_10"
file_coords = ""
nlayer = 11

# load vertices and faces
v_ref, f_ref = read_geometry(file_middle)
v_white, _ = read_geometry(file_white)
v_pial, _ = read_geometry(file_pial)
cords = load_coords(file_coords)

A = Layer(coords, v_ref, f_ref, v_white, v_pial)
bla = A.generate_layers(5)
```

## References
<a id="1">[1]</a> de Hollander, G. D., van der Zwaag, W., Qian, C., Zhang, P. &  Knapen, T. Ultra-high field fMRI reveals origins of feedforward and feedback activity within laminae of human ocular dominance columns. *NeuroImage* **228** (2021). 

<a id="2">[2]</a> Nasr, S., Polimeni, J. R. & Tootell R. B. H. Interdigitated Color- and Disparity-Selective Columns within Human Visual Cortical Areas V2 and V3. *J. Neurosci.* **36,** 1841&ndash;1857 (2016).


## Contact
If you have questions, problems or suggestions regarding the column_filter package, please feel free to contact [me](mailto:daniel.haenelt@gmail.com).
