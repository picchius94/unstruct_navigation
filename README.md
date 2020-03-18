# Mobile Robot Navigation in Unstructured Terrains 
This repo contains three main utilities:
- Random generation of unstructured environments using the OpenSimplex algorithm
- Geometric traversability analysis for wheel robots
- A* on a lattice space path planning algorithm

## 1. Natural terrain generation with OpenSimplex
In `terrain_generator.py` the Opensimplex Python API is used along with three filtering techniques to render realistic terrains. Fist, an `OpenSimplex_Map` object has to be initialised with:
- *map_size*: size in metres of the squared-map from *-map_size/2* to *map_size/2*
- *discr*: discretization in metres of each map cell
- *terrain_type*: 5 options (`mountain_crater`, `smooth`, `rough`, `wavy`, `scattered_sharp`)
