# Mobile Robot Navigation in Unstructured Terrains 
This repo contains three main utilities:
- Random generation of unstructured environments using the OpenSimplex algorithm
- Geometric traversability analysis for wheel robots
- A* on a lattice space path planning algorithm

## 1. Natural terrain generation with OpenSimplex
The [Opensimplex Python API](https://github:com/lmas/opensimplex) is used along with some filtering techniques to render realistic terrains.

This is done by instantiate a `terrain_generator.OpenSimplex_Map` object with the following parameters:
- *map_size*: size in metres of the squared-map from *-map_size/2* to *map_size/2*
- *discr*: discretization in metres of each map cell
- *terrain_type*: 5 options are defined (`mountain_crater`, `smooth`, `rough`, `wavy`, `scattered_sharp`), which differently set the parameters of the filters (once the object has been instantiated it is also possible to change each parameter individually) 
