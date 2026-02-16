# energyplus-geometry

Single-file Python library for EnergyPlus geometry generation (`ep_geometry.py`).

## GitHub Action usage

This repository can be used directly as a composite action:

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: mpaulus/energyplus-geometry@v1
  - run: python your_script.py
```

The action adds its own repo path to `PYTHONPATH` for later steps in the same job.
That makes imports like this work in downstream scripts:

```python
from ep_geometry import P, Z
```

Notes:
- Python must already be available on the runner.
- `ep_geometry.py` imports `shapely`, so install dependencies in the calling workflow if needed.

## Library example

```python
from ep_geometry import P, p2w, Z, all_wall_data_from_zones, group_walls, wall_analyze, print_idf_wall, ceiling_split

p1 = P(0, 0)
p2 = p1.right(3)
p3 = p2.up(4)
p4 = p3.left(2)

walls = p2w([p1, p2, p3, p4])
x_origin = 0
y_origin = 10
z_origin = 0
zone = Z("Zone", walls, x_origin, y_origin, z_origin, "Space Name")

wall_data = all_wall_data_from_zones(zones, "Additional Text")
grouped_walls = group_walls(wall_data)
matches = wall_analyze(grouped_walls)

for match in matches:
    wall_surface = print_idf_wall(match, 30, 0, "Ext Wall Construction")
    print(wall_surface)

roof_surfaces = ceiling_split(zones, [], 30, "Ceiling", "Roof", {}, {})
for roof in roof_surfaces:
    print(roof)

ffactor_val = 1.26
for zone in floor_1_zones:
    ffactor_construction = zone.floor_ffactor_construction(ffactor_val, matches)
    print(ffactor_construction, end="\n")

for zone in floor_1_zones:
    print(zone.idf_floor(0, zone.floor_ffactor_construction_name()), end="\n")
```
