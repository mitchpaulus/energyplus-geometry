```
from ep_geometry import P, p2w, Z

p1 = P(0, 0)
p2 = p1.right(3)
p3 = p2.up(4)
p4 = p3.left(2)

walls = p2w([p1, p2, p3, p4])
x_origin = 0
y_origin = 10
z_origin = 0
zone = Z("Zone", walls, x_origin, y_origin, z_origin, "Space Name")

```
