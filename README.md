# harm_pythontools:
Python and related bash scripts and C code to analyze HARM data. Now in python3!

## Important functions:
The most important function so far is `render_and_load_iso_points()` in `__init_simple__.py`. It takes the fieldline number as input and will display an "isosurface" comprised of points, with an interactive slider that lets you filter out density. It will also display the number of the fieldline being rendered.

![Pyvista](/README_images/pyvista.png)


You can also render and then load by running something like this:
```python
coords,data = render_iso_full_density(fnumber)
load_point_plot(coords,data)
```
This first function returns the cartesian coordinates and the density, to be fed into the second function.

We also tried testing things out in YT-project for a while, which sadly turned out to be incompatible with our semi-strucutred spherical grid. In the event that YT does become viable for volume rendering with our data, we have kept the YT-related functions in `__init__simple__.py`. To use this, you can similarly run the following lines:

```python
r,theta,phi = make_simplified_array('fieldline#####.bin')
ds = load_simplified_array(r,theta,phi)
```
From there you can do all kinds of object oriented operations on `ds`. We found use in making sliceplots buy running a line such as 
```python
import yt
slc = yt.SlicePlot(ds, 'theta', 'density')
slc.show() # if on a jupyter notebook
slc.save('screenshot_name') # otherwise
```
Which would render a plot like this one:


![YT_example](/README_images/yt_example.png)

From there, you can do all kinds of object-oriented operations to alter the plot, which is detailed in [YT project's documentation](https://yt-project.org/doc/reference/api/yt.visualization.profile_plotter.html).
