# Mobile Robot Navigation in Unstructured Terrains 
This repo contains three main utilities:
- Random generation of unstructured environments using the OpenSimplex algorithm
- Geometric traversability analysis for wheel robots
- A* on a lattice space path planning algorithm

## 1. Natural terrain generation with OpenSimplex
The [Opensimplex Python API](https://github:com/lmas/opensimplex) is used along with some filtering techniques to render realistic terrains.

This is done by instantiate a `terrain_generator.OpenSimplex_Map` object with the following parameters:
- __*map_size*__: size in metres of the squared-map from *-map_size/2* to *map_size/2*
- __*discr*__: discretization in metres of each map cell
- __*terrain_type*__: 5 options are defined (`mountain_crater`, `smooth`, `rough`, `wavy`, `scattered_sharp`), which differently set the parameters of the filters

Then, calling the `sample_generator` method a new sample is generated. Here, [an example](Images_example/Figure_2):

```python
import terrain_generator as tg
terrain = tg.OpenSimplexMap(map_size, discr, "wavy")
terrain.sample_generator(plot=True)
```

The map matrix can be accessed by calling  `terrain.Z`.

Upon initialization, it is also possible to change each filter parameter individually, for example:

```python
terrain.perc_obstacles = 0.12
terrain.max_slope_variation = 0.5
```
The full list of parameters can be found in the class definition.
