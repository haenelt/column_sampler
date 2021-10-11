# Column sampler

<p align="center">
  <img src="https://github.com/haenelt/column_sampler/blob/main/img/example.gif?raw=true" width=75% height=75% alt="Illustration of GBB"/>
</p>

Ultrahigh field functional magnetic resonance imaging (fMRI) enables the mapping of cortical layers and columns in living humans. However, fMRI still suffers from various noise sources which challenges the interpretation of the measured fMRI signal. An accurate characterization of the BOLD point-spread function (PSF) is therefore a prerequisite for further understanding of the measured signal.

This package provides functionality to segment out single cortical columns mapped on a surface mesh. This can be used to analyze the edge response at column borders to study the BOLD PSF across cortical depth similar to [[1]](#1).

The figure above shows an example mesh which defines lines orthogonal to an oriented cortical columns. First, a path is manually drawn on a reference surface mesh. Then, lines are computed at different cortical depths which are oriented orthogonal to this path. Finally, functional contrasts can be sampled onto these lines to analyze the edge response around column borders at different cortical depths.

## Installation
I recommend to use `Miniconda` to create a new python environment with `python=3.8`. To install the package, clone the repository and run the following line from the directory in which the repository was cloned

```shell
python setup.py install
```

After the installation you should be able to import the package with `import column_sampler`.

## Some snippets

### Planar Mesh
```python
from nibabel.freesurfer.io import read_geometry
from column_sampler.io import coords_to_mesh
from column_sampler.mesh import PlanarMesh

surf_in = ""  # reference surface in freesurfer format
plane_out = ""  # file name of output mesh
ind = []  # list of vertex indices

v, f = read_geometry(surf_in)  # read vertices and faces
plane = PlanarMesh(v, f, ind)
plane.save_line(line_out)  # save line as MGH overlay
coords_to_mesh(plane_out, plane.line_coordinates)  # save plane as mesh
```

### Curved Mesh
```python
from nibabel.freesurfer.io import read_geometry
from column_sampler.io import save_coords, coords_to_mesh
from column_sampler.mesh import CurvedMesh

surf_in = ""  # reference surface in freesurfer format
curv_out = ""  # file name of output mesh
coords_out = ""  # file name of output coordinate array
ind = []  # list of vertex indices

v, f = read_geometry(surf_in)
curv = CurvedMesh(v, f, ind)
coords = curv.project_coordinates_sequence()
coords_to_mesh(curv_out, coords)
save_coords(coords_out, coords)
```

### Layering
```python
from nibabel.freesurfer.io import read_geometry
from column_sampler.layer import Layer

# layers
file_white = ""  # file name of white surface
file_middle = ""  # file name of reference surface
file_pial = ""  # file name of pial surface
file_coords = ""  # file name of input coordinate array
nlayer = 11  # number of defined layers

# load vertices and faces
v_ref, f_ref = read_geometry(file_middle)
v_white, _ = read_geometry(file_white)
v_pial, _ = read_geometry(file_pial)
cords = load_coords(file_coords)

layer = Layer(coords, v_ref, f_ref, v_white, v_pial).generate_layers(nlayer)
```

## References
<a id="1">[1]</a> Fracasso, A., Dumoulin, S. O., Petridou, N. Point-spread function of the BOLD response across columns and cortical depth in human extra-striate cortex. *Prog. Neurobiol.* **202** (2021).

## Contact
If you have questions, problems or suggestions regarding the column_filter package, please feel free to contact [me](mailto:daniel.haenelt@gmail.com).
