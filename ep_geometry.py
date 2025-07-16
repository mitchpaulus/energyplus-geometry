#!/usr/bin/python3

import sys
import random
import uuid
import shapely

from typing import Union

class WindowData:
    def __init__(self, surface, count, width_ft, surface_length_ft) -> None:
        self.surface = surface
        self.count = count
        self.width_ft = width_ft
        self.surface_length_ft = surface_length_ft

    def __str__(self) -> str:
        return f"{self.surface}\t{self.count}\t{self.width_ft}\t{self.surface_length_ft}"

    def __repr__(self) -> str:
        return self.__str__()


class WindowSurface:
    def __init__(self) -> None:
        self._name = ""
        self._construction_name = ""
        self._building_surface_name = ""
        self._frame_and_divider_name = ""
        self._multiplier = 1.0
        self._starting_x = None
        self._starting_z = None
        self._length = None
        self._height = None

    def __str__(self) -> str:
        if self._length is None or self._height is None or self._starting_x is None or self._starting_z is None:
            raise ValueError("Length, Height, Starting X, and Starting Z must be set for WindowSurface")

        if self._name == "":
            raise ValueError("Name must be set for WindowSurface")

        if self._construction_name == "":
            raise ValueError("Construction Name must be set for WindowSurface")

        if self._building_surface_name == "":
            raise ValueError("Building Surface Name must be set for WindowSurface")

        lines = []
        lines.append("Window,\n")
        lines.append(f"  {self._name}, !- Name\n")
        lines.append(f"  {self._construction_name}, !- Construction Name\n")
        lines.append(f"  {self._building_surface_name}, !- Building Surface Name\n")
        lines.append(f"  {self._frame_and_divider_name}, !- Frame and Divider Name\n")
        lines.append(f"  {self._multiplier}, !- Multiplier\n")
        lines.append(f"  {self._starting_x}, !- Starting X Coordinate ({self._starting_x * 3.2808399:.1f} ft)\n")
        lines.append(f"  {self._starting_z}, !- Starting Z Coordinate ({self._starting_z * 3.2808399:.1f} ft) \n")
        lines.append(f"  {self._length}, !- Length ({self._length * 3.2808399:.1f} ft) \n")
        lines.append(f"  {self._height}; !- Height ({self._height * 3.2808399:.1f} ft)\n")
        return "".join(lines)

    def name(self, name):
        self._name = name
        return self

    def construction_name(self, construction_name):
        self._construction_name = construction_name
        return self

    def building_surface_name(self, building_surface_name):
        self._building_surface_name = building_surface_name
        return self

    def frame_and_divider_name(self, frame_and_divider_name):
        self._frame_and_divider_name = frame_and_divider_name
        return self

    def multiplier(self, multiplier):
        self._multiplier = multiplier
        return self

    def starting_x(self, starting_x):
        self._starting_x = starting_x
        return self

    def starting_z(self, starting_z):
        self._starting_z = starting_z
        return self

    def length(self, length):
        """Aka width"""
        self._length = length
        return self

    def height(self, height):
        self._height = height
        return self

# Door {{{

class Door:
    def __init__(self) -> None:
        self._name = ""
        self._construction_name = ""
        self._building_surface_name = ""
        self._multiplier = 1.0
        self._starting_x = 0.0
        self._starting_z = 0.0
        self._length = None
        self._height = None

    def __str__(self) -> str:
        if self._length is None or self._height is None:
            raise ValueError("Length and Height must be set for Door")
        if self._name == "":
            raise ValueError("Name must be set for Door")
        if self._construction_name == "":
            raise ValueError("Construction Name must be set for Door")
        if self._building_surface_name == "":
            raise ValueError("Building Surface Name must be set for Door")

        lines = []
        lines.append("Door,\n")
        lines.append(f"  {self._name}, !- Name\n")
        lines.append(f"  {self._construction_name}, !- Construction Name\n")
        lines.append(f"  {self._building_surface_name}, !- Building Surface Name\n")
        lines.append(f"  {self._multiplier}, !- Multiplier\n")
        lines.append(f"  {self._starting_x}, !- Starting X Coordinate ({self._starting_x * 3.2808399:.1f} ft)\n")
        lines.append(f"  {self._starting_z}, !- Starting Z Coordinate ({self._starting_z * 3.2808399:.1f} ft) \n")
        lines.append(f"  {self._length}, !- Length ({self._length * 3.2808399:.1f} ft) \n")
        lines.append(f"  {self._height}; !- Height ({self._height * 3.2808399:.1f} ft)\n")
        return "".join(lines)

    def name(self, name):
        self._name = name
        return self

    def construction_name(self, construction_name):
        self._construction_name = construction_name
        return self

    def building_surface_name(self, building_surface_name):
        self._building_surface_name = building_surface_name
        return self

    def multiplier(self, multiplier):
        self._multiplier = multiplier
        return self

    def starting_x(self, starting_x):
        self._starting_x = starting_x
        return self

    def starting_x_ft(self, starting_x_ft):
        self._starting_x = starting_x_ft * 0.3048
        return self

    def starting_z(self, starting_z):
        self._starting_z = starting_z
        return self

    def starting_z_ft(self, starting_z_ft):
        self._starting_z = starting_z_ft * 0.3048
        return self

    def length(self, length):
        self._length = length
        return self

    def length_ft(self, length_ft):
        self._length = length_ft * 0.3048
        return self

    def height(self, height):
        self._height = height
        return self

    def height_ft(self, height_ft):
        self._height = height_ft * 0.3048
        return self
# }}}

# GlazedDoor {{{
class GlazedDoor:
    def __init__(self) -> None:
        self._name = ""
        self._construction_name = ""
        self._building_surface_name = ""
        self._frame_and_divider_name = ""
        self._multiplier = 1.0
        self._starting_x = 0.0
        self._starting_z = 0.0
        self._length = None
        self._height = None

    def __str__(self) -> str:
        if self._length is None or self._height is None:
            raise ValueError("Length and Height must be set for GlazedDoor")
        if self._name == "":
            raise ValueError("Name must be set for GlazedDoor")
        if self._construction_name == "":
            raise ValueError("Construction Name must be set for GlazedDoor")
        if self._building_surface_name == "":
            raise ValueError("Building Surface Name must be set for GlazedDoor")

        lines = []
        lines.append("GlazedDoor,\n")
        lines.append(f"  {self._name}, !- Name\n")
        lines.append(f"  {self._construction_name}, !- Construction Name\n")
        lines.append(f"  {self._building_surface_name}, !- Building Surface Name\n")
        lines.append(f"  {self._frame_and_divider_name}, !- Frame and Divider Name\n")
        lines.append(f"  {self._multiplier}, !- Multiplier\n")
        lines.append(f"  {self._starting_x}, !- Starting X Coordinate ({self._starting_x * 3.2808399:.1f} ft)\n")
        lines.append(f"  {self._starting_z}, !- Starting Z Coordinate ({self._starting_z * 3.2808399:.1f} ft) \n")
        lines.append(f"  {self._length}, !- Length ({self._length * 3.2808399:.1f} ft) \n")
        lines.append(f"  {self._height}; !- Height ({self._height * 3.2808399:.1f} ft)\n")

        return "".join(lines)

    # Fluent Properties {{{
    def name(self, name):
        self._name = name
        return self

    def construction_name(self, construction_name):
        self._construction_name = construction_name
        return self

    def building_surface_name(self, building_surface_name):
        self._building_surface_name = building_surface_name
        return self

    def frame_and_divider_name(self, frame_and_divider_name):
        self._frame_and_divider_name = frame_and_divider_name
        return self

    def multiplier(self, multiplier):
        self._multiplier = multiplier
        return self

    def starting_x(self, starting_x):
        self._starting_x = starting_x
        return self

    def starting_x_ft(self, starting_x_ft):
        self._starting_x = starting_x_ft * 0.3048
        return self

    def starting_z(self, starting_z):
        self._starting_z = starting_z
        return self

    def starting_z_ft(self, starting_z_ft):
        self._starting_z = starting_z_ft * 0.3048
        return self

    def length(self, length):
        self._length = length
        return self

    def length_ft(self, length_ft):
        self._length = length_ft * 0.3048
        return self

    def height(self, height):
        self._height = height
        return self

    def height_ft(self, height_ft):
        self._height = height_ft * 0.3048
        return self

    # }}}

# }}}

def spacing_around2(total_width, window_width, n):
    l_wall = (total_width - window_width * n)
    l_spaces = l_wall / (n + 1)
    xs = []
    for i in range(0, n):
        xs.append(l_spaces + i * (l_spaces + window_width))
    return xs

def spacing_around(l, w, n):
    xs = []

    for i in range(1, n+1):
        xs.append(i * (l/(n+1)) - (w/2))

    return xs

def center_x(total_width, window_width):
    return (total_width - window_width) / 2

class FfactorConstruction:
    def __init__(self) -> None:
        self._name = ""
        self._ffactor = ""
        self._area = ""
        self._perimeter_exposed = ""

    def name(self, name):
        self._name = name
        return self

    def ffactor(self, ffactor):
        self._ffactor = ffactor
        return self

    def area_m2(self, area_m2):
        self._area = area_m2
        return self

    def perimeter_exposed(self, perimeter_exposed):
        self._perimeter_exposed = perimeter_exposed
        return self

    def __str__(self) -> str:
        if self._name == "":
            raise ValueError("Name must be set for FfactorConstruction")
        if self._ffactor == "":
            raise ValueError("Ffactor must be set for FfactorConstruction")
        if self._area == "":
            raise ValueError("Area must be set for FfactorConstruction")
        if self._perimeter_exposed == "":
            raise ValueError("Perimeter Exposed must be set for FfactorConstruction")

        lines = []
        lines.append("Construction:FfactorGroundFloor,\n")
        lines.append(f"  {self._name}, !- Name\n")
        lines.append(f"  {self._ffactor}, !- Ffactor {{W/m-K}}\n")
        lines.append(f"  {self._area}, !- Area {{m2}}\n")
        lines.append(f"  {self._perimeter_exposed}; !- Perimeter Exposed {{m}}\n")
        return "".join(lines)

class IdfBuildingSurface:
    def __init__(self) -> None:
        self._name = ""
        self._surface_type = ""
        self._construction_name = ""
        self._zone_name = ""
        self._space_name = ""
        self._outside_boundary_condition = ""
        self._outside_boundary_condition_object = ""
        self._sun_exposure = ""
        self._wind_exposure = ""
        self._vertices = []

    def __str__(self) -> str:
        lines = []
        lines.append("BuildingSurface:Detailed,\n")
        lines.append(f"  {self._name},!- Name\n")
        lines.append(f"  {self._surface_type}, !- Surface Type\n")
        lines.append(f"  {self._construction_name}, !- Construction Name\n")
        lines.append(f"  {self._zone_name}, !- Zone Name\n")
        lines.append(f"  {self._space_name}, !- Space Name\n")
        lines.append(f"  {self._outside_boundary_condition}, !- Outside Boundary Condition\n")
        lines.append(f"  {self._outside_boundary_condition_object}, !- Outside Boundary Condition Object\n")
        lines.append(f"  {self._sun_exposure}, !- Sun Exposure\n")
        lines.append(f"  {self._wind_exposure}, !- Wind Exposure\n")
        lines.append(f"  , !- View Factor to Ground\n")
        lines.append(f"  , !- Number of Vertices\n")

        for i, vertex in enumerate(self._vertices, 1):
            if i == len(self._vertices):
                lines.append(f"  {vertex[0]:.6f}, {vertex[1]:.6f}, {vertex[2]:.6f}; !- X,Y,Z ==> Vertex {i}\n")
            else:
                lines.append(f"  {vertex[0]:.6f}, {vertex[1]:.6f}, {vertex[2]:.6f}, !- X,Y,Z ==> Vertex {i}\n")
        return "".join(lines)


    def wall_length_m(self):
        # Check if first and second vertex have same z
        if abs(self._vertices[0][2] - self._vertices[1][2]) > 1e-6:
            raise ValueError("First and second vertex must have same z value")

        x1 = self._vertices[0][0]
        y1 = self._vertices[0][1]
        x2 = self._vertices[1][0]
        y2 = self._vertices[1][1]
        return ((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)) ** 0.5

    def wall_length_ft(self):
        return self.wall_length_m() / 0.3048

    def name(self, name):
        self._name = name
        return self

    def surface_type(self, surface_type):
        self._surface_type = surface_type
        return self

    def construction_name(self, construction_name):
        self._construction_name = construction_name
        return self

    def zone_name(self, zone_name):
        self._zone_name = zone_name
        return self

    def space_name(self, space_name):
        self._space_name = space_name
        return self

    def outside_boundary_condition(self, outside_boundary_condition):
        self._outside_boundary_condition = outside_boundary_condition
        if outside_boundary_condition == "Outdoors":
            self._sun_exposure = "SunExposed"
            self._wind_exposure = "WindExposed"
            self._outside_boundary_condition_object = ""
        else:
            self._sun_exposure = "NoSun"
            self._wind_exposure = "NoWind"
        return self

    def outside_boundary_condition_object(self, outside_boundary_condition_object):
        self._outside_boundary_condition_object = outside_boundary_condition_object
        return self

    def sun_exposure(self, sun_exposure):
        self._sun_exposure = sun_exposure
        return self

    def wind_exposure(self, wind_exposure):
        self._wind_exposure = wind_exposure
        return self

    def vertices(self, vertices):
        self._vertices = vertices
        return self


class UuidGenerator:
    def __init__(self, seed) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def uuid(self):
        return uuid.UUID(int=self.rng.getrandbits(128))

def line_prop(x1, y1, x2, y2):
    # l = A1 A2 = <y1 - y2 : x2 - x1 : x1 y2 - x2 y1>
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return (a, b, c)

def spread(l1, l2):
    # s(l1, l2) = (a1 b2 - a2 b1)² / (a1² + b1²)(a2² + b2²)
    a1 = l1[0]
    b1 = l1[1]
    a2 = l2[0]
    b2 = l2[1]
    return (a1 * b2 - a2 * b1)*(a1 * b2 - a2 * b1) / ((a1*a1 + b1*b1) * (a2*a2 + b2*b2))

def line_spread(l1):
    # Take spread vs. horizontal line, which is proportion: 0, 1, 0
    l2 = [0, 1, 0]
    return spread(l1, l2)

# https://stackoverflow.com/a/16691908/5932184
def overlap(p11, p12, p21, p22):
    min1 = min(p11, p12)
    max1 = max(p11, p12)
    min2 = min(p21, p22)
    max2 = max(p21, p22)

    x1 = max(min1, min2)
    x2 = min(max1, max2)

    return (max(0, x2 - x1), x1, x2)


def p_2d_to_1d(*args):
    if len(args) == 1 and isinstance(args[0], P):
        p = args[0]
        x = p.x
        y = p.y
    elif len(args) == 2:
        x, y = args
    else:
        raise ValueError('Invalid arguments to p_2d_to_1d')

    dist = (x*x + y*y)**0.5
    if x < 0:
        return -dist
    else:
        return dist

class Wall:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, int_ext: str) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.int_ext = int_ext

    def line(self):
        """Use line definition from Divine Proportions"""
        return line_prop(self.x1, self.y1, self.x2, self.y2)

    def __str__(self) -> str:
        return "\t".join([str(self.x1), str(self.y1), str(self.x2), str(self.y2), self.int_ext])

    def __repr__(self):
        return self.__str__()

    def length(self):
        return ((self.x2 - self.x1)*(self.x2 - self.x1) + (self.y2 - self.y1)*(self.y2 - self.y1))**0.5

    def __eq__(self, other):
        # 3 Conditions: a1 b2 - a2 b1 = 0, b1 c2 - b2 c1 = 0, c1 a2 - c2 a1 = 0
        l1 = self.line()
        l2 = other.line()
        return line_equals(l1, l2)

def line_equals(l1, l2):
    tol = 1e-6
    # 3 Conditions: a1 b2 - a2 b1 = 0, b1 c2 - b2 c1 = 0, c1 a2 - c2 a1 = 0
    cond1 = abs(l1[0] * l2[1] - l1[1] * l2[0]) < tol
    if not cond1:
        return False
    cond2 = abs(l1[1] * l2[2] - l1[2] * l2[1]) < tol
    if not cond2:
        return False
    cond3 = abs(l1[2] * l2[0] - l1[0] * l2[2]) < tol
    if not cond3:
        return False
    return True

class NamedWall:
    """
    This is a wall (2d line) with a name and a zone/space
    """
    def __init__(self, zone: str, name: str, wall: Wall, space: str) -> None:
        self.zone = zone
        self.name = name
        self.wall = wall
        self.space = space

    def __str__(self) -> str:
        return f"Zone: {self.zone} Space: {self.space}, {self.name}: {self.wall}"

    def __repr__(self) -> str:
        return self.__str__()

    def clone(self):
        return NamedWall(self.zone, self.name, self.wall, self.space)

    def same_space_zone(self, other):
        if self.space.strip() == '' and other.space.strip() == '':
            return self.zone == other.zone
        else:
            return self.space == other.space and self.zone == other.zone

class P_2d_to_1d:
    def __init__(self, *args) -> None:
        if len(args) == 1 and isinstance(args[0], Wall):
            self.x1 = args[0].x1
            self.y1 = args[0].y1
            self.x2 = args[0].x2
            self.y2 = args[0].y2
        if len(args) == 1 and isinstance(args[0], NamedWall):
            self.x1 = args[0].wall.x1
            self.y1 = args[0].wall.y1
            self.x2 = args[0].wall.x2
            self.y2 = args[0].wall.y2
        else:
            raise ValueError('Invalid arguments to p_2d_to_1d')

    def to_1d(self, x, y):
        # First, force through origin. Shift y by adding c/b

        # Vertical line
        if self.x1 - self.x2 != 0:
            return (x - self.x2) / (self.x1 - self.x2)
        else:
            return (y - self.y2) / (self.y1 - self.y2)

    def to_2d(self, t):
        return t * self.x1 + (1 - t) * self.x2, t * self.y1 + (1 - t) * self.y2

class Z:
    def __init__(self, name: str, walls: list[Wall], origin_x_ft, origin_y_ft, origin_z_ft, space_name: str) -> None:
        self.name = name
        self.walls = walls
        self.origin_x_ft = origin_x_ft
        self.origin_y_ft = origin_y_ft
        self.origin_z_ft = origin_z_ft
        self.space_name = space_name


    def name_for_ref(self):
        if self.space_name.strip() == '':
            return self.name
        else:
            return self.space_name

    def __str__(self) -> str:
        points = []
        i = 0
        point = P(self.walls[i].x1, self.walls[i].y1)
        points.append(point)
        prev_point = point

        while i < len(self.walls):
            new_point = P(self.walls[i].x2, self.walls[i].y2)
            if new_point == prev_point:
                i += 1
                continue

            points.append(new_point)
            prev_point = new_point
            i += 1

        fields = [self.name, self.space_name]
        for point in points:
            fields.append(str(point.x))
            fields.append(str(point.y))

        return "\t".join(fields)

    def shapely_poly(self):
        return shapely.geometry.Polygon([(wall.x1 + self.origin_x_ft, wall.y1 + self.origin_y_ft) for wall in self.walls])

    def wall_data(self, start_num = 1, add_text = '') -> list[NamedWall]:
        all_walls = []
        for i, wall in enumerate(self.walls, start_num):
            if add_text != '' and add_text is not None:
                wall_name = f"{self.name_for_ref()} {add_text} Wall {i}"
            else:
                wall_name = f"{self.name_for_ref()} Wall {i}"
            new_wall = self.adjust_wall_coords(wall)
            named_wall = NamedWall(self.name, wall_name, new_wall, self.space_name)
            all_walls.append(named_wall)
        return all_walls


    def adjust_wall_coords(self, input_wall):
        # Adjust wall coordinates to be relative to the zone origin
        return Wall(input_wall.x1 + self.origin_x_ft, input_wall.y1 + self.origin_y_ft, input_wall.x2 + self.origin_x_ft, input_wall.y2 + self.origin_y_ft, input_wall.int_ext)


    def idf_floor(self, z_ft, construction) -> IdfBuildingSurface:
        floor_surface = IdfBuildingSurface()
        floor_surface.name(f"{self.name_for_ref()} Floor").surface_type("Floor").construction_name(construction)
        floor_surface.zone_name(self.name).outside_boundary_condition("Ground").outside_boundary_condition_object("")
        floor_surface.space_name(self.space_name)
        floor_surface.sun_exposure("NoSun").wind_exposure("NoWind")

        vertices = []
        z = z_ft * 0.3048
        # Reversed needed to correct the orientation.
        for wall in reversed(self.walls):
            wall = self.adjust_wall_coords(wall)


            vertices.append((wall.x1 * 0.3048 , wall.y1 * 0.3048, z))

        # Remove collinear points
        i = 0
        while i < len(vertices):
            x1 = vertices[(i-1) % len(vertices)][0]
            y1 = vertices[(i-1) % len(vertices)][1]
            x2 = vertices[i][0]
            y2 = vertices[i][1]
            x3 = vertices[(i+1) % len(vertices)][0]
            y3 = vertices[(i+1) % len(vertices)][1]

            # Check for collinear points
            if abs(x1 * y2 - x1 * y3 + x2 * y3 - x3 * y2 + x3 * y1 - x2 * y1) < 1e-6:
                vertices.pop(i)
            else:
                i += 1

        floor_surface.vertices(vertices)
        return floor_surface


    def ffactor_construction(self,ffactor):
        floor_surface = FfactorConstruction()

        area = self.shapely_poly().area # ft^2
        area_m2 = area / 10.7639

        perimeter = 0
        for w in self.walls:
            if w.int_ext == 'e':
                perimeter += w.length()

        perimeter_m = perimeter * 0.3048

        floor_surface.name(f"{self.name_for_ref()} Floor FFactor").ffactor(ffactor).area_m2(area_m2).perimeter_exposed(perimeter_m)
        return floor_surface

    def exterior_ceiling_outdoors(self, z_ft, construction) -> list[IdfBuildingSurface]:
        z = z_ft * 0.3048
        vertices = []
        for wall in self.walls:
            wall = self.adjust_wall_coords(wall)
            # Check for collinear points
            if len(vertices) >= 2:
                x1 = vertices[-2][0]
                y1 = vertices[-2][1]
                x2 = vertices[-1][0]
                y2 = vertices[-1][1]
                x3 = wall.x1 * 0.3048
                y3 = wall.y1 * 0.3048

                # x1 y2 - x1 y3 + x2 y3 - x3 y2 + x3 y1 - x2 y1 = 0
                if abs(x1 * y2 - x1 * y3 + x2 * y3 - x3 * y2 + x3 * y1 - x2 * y1) < 1e-6:
                    # Remove the middle point, or last current point on the list.
                    vertices.pop()

            vertices.append((wall.x1 * 0.3048, wall.y1 * 0.3048, z))

        if not is_convex(vertices):
            surfaces = []
            triangles = earclip(vertices)
            triangle_num = 1
            for triangle in triangles:
                ceiling_surface = IdfBuildingSurface()
                ceiling_surface.name(f"{self.name_for_ref()} Ceiling {triangle_num}").surface_type("Roof").construction_name(construction)
                ceiling_surface.zone_name(self.name).outside_boundary_condition("Outdoors").outside_boundary_condition_object("").sun_exposure("SunExposed").wind_exposure("WindExposed")
                ceiling_surface.space_name(self.space_name)
                ceiling_surface.vertices(triangle)
                triangle_num += 1
                surfaces.append(ceiling_surface)

            return surfaces

        else:
            ceiling_surface = IdfBuildingSurface()
            ceiling_surface.name(f"{self.name_for_ref()} Ceiling").surface_type("Roof").construction_name(construction)
            ceiling_surface.zone_name(self.name).outside_boundary_condition("Outdoors").outside_boundary_condition_object("").sun_exposure("SunExposed").wind_exposure("WindExposed")
            ceiling_surface.space_name(self.space_name)
            ceiling_surface.vertices(vertices)

            return [ceiling_surface]

def group_walls(wall_list: list[NamedWall]) -> list[list[NamedWall]]:
    """This function groups wall that are on the same line in the flat 2-D plane"""
    indexes_used = set()
    groups = []

    for i, wall in enumerate(wall_list):
        if i in indexes_used:
            continue

        group = [wall]
        indexes_used.add(i)

        for j, other_wall in enumerate(wall_list):
            if j in indexes_used:
                continue

            if wall.wall == other_wall.wall:
                group.append(other_wall)
                indexes_used.add(j)

        groups.append(group)

    return groups

def right_edge(wall_list: list[Wall]):
     max_x1 = max(wall.x1 for wall in wall_list)
     max_x2 = max(wall.x2 for wall in wall_list)
     return max(max_x1, max_x2)

def left_edge(wall_list: list[Wall]):
    min_x1 = min(wall.x1 for wall in wall_list)
    min_x2 = min(wall.x2 for wall in wall_list)
    return min(min_x1, min_x2)

def cross_product(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

def signed_area(verticies):
    area = 0
    for i in range(len(verticies)):
        j = (i + 1) % len(verticies)
        area += verticies[i][0] * verticies[j][1] - verticies[j][0] * verticies[i][1]
    return area / 2

def signed_area_from_walls(walls: list[Wall]):
    verticies = [(wall.x1, wall.y1) for wall in walls]
    return signed_area(verticies)

def is_convex(verticies):

    sign = 0
    for i in range(len(verticies)):
        j = (i + 1) % len(verticies)
        k = (i + 2) % len(verticies)
        z = cross_product((verticies[j][0] - verticies[i][0], verticies[j][1] - verticies[i][1]), (verticies[k][0] - verticies[j][0], verticies[k][1] - verticies[j][1]))
        if z < 0:
            sign += -1
        else:
            sign += 1

        if abs(sign) - 1 != i:
            return False
    return True

def earclip(polygon):
    def vector(p1, p2):
        return p2[0]-p1[0], p2[1]-p1[1]

    def cross(v1, v2):
        return v1[0]*v2[1] - v1[1]*v2[0]

    def is_inside(p, a, b, c):
        ap, bp, cp = vector(a, p), vector(b, p), vector(c, p)
        cross_1 = cross(vector(a, b), ap)
        if cross_1 < 0:
            return False
        cross_2 = cross(vector(b, c), bp)
        if cross_2 < 0:
            return False
        cross_3 = cross(vector(c, a), cp)
        if cross_3 < 0:
            return False

        return True

    def is_ear(i, all_points, available_indices):
        ai, bi, ci = available_indices[i], available_indices[(i + 1) % len(available_indices)], available_indices[(i + 2) % len(available_indices)]
        a, b, c = all_points[ai], all_points[bi], all_points[ci]

        # First check if interior angle is < 180
        ab, bc = vector(a, b), vector(a, c)
        if cross(ab, bc) < 0:
            return False

        for j in available_indices:
            if j in (ai, bi, ci):
                continue
            if is_inside(all_points[j], a, b, c):
                return False

        return True

    available_indices = list(range(len(polygon)))

    triangles = []

    while len(available_indices) > 3:
        for i in range(len(available_indices)):
            if is_ear(i, polygon, available_indices):
                ai, bi, ci = available_indices[i], available_indices[(i + 1) % len(available_indices)], available_indices[(i + 2) % len(available_indices)]
                a, b, c = polygon[ai], polygon[bi], polygon[ci]
                triangles.append((a, b, c))
                available_indices.pop((i + 1) % len(available_indices))
                break
        else:
            raise Exception("No ear found, by mathematical proof this is not OK.")

    ai, bi, ci = available_indices[0], available_indices[1], available_indices[2]
    a, b, c = polygon[ai], polygon[bi], polygon[ci]
    triangles.append((a, b, c))

    return triangles


def ceiling_split(lower_level_zones: list[Z], upper_level_zones: list[Z], z_ft: float, construction_name: str, roof_construction: str, ceiling_counts: dict[str, int], floor_counts: dict[str, int]) -> list[IdfBuildingSurface]:
    idf_surfaces = []
    # Mapping from zone name to current counts
    #  ceiling_counts = {}
    #  floor_counts = {}
    for lower_level_zone in lower_level_zones:
        lower_level_poly = lower_level_zone.shapely_poly()
        for upper_level_zone in upper_level_zones:
            upper_level_poly = upper_level_zone.shapely_poly()
            if lower_level_poly.intersects(upper_level_poly):
                intersection = lower_level_poly.intersection(upper_level_poly)
                if intersection.area > 0.0001:
                    lower_level_poly = lower_level_poly.difference(intersection)
                    upper_level_poly = upper_level_poly.difference(intersection)

                    # This allows for multi-story zones.
                    if lower_level_zone.name == upper_level_zone.name:
                        continue

                    # Counter-clockwise
                    intersection = shapely.geometry.polygon.orient(intersection, sign=1.0)
                    verticies = []
                    for point in intersection.exterior.coords:
                        # Covert to m
                        x = point[0] * 0.3048
                        y = point[1] * 0.3048

                        # Check for coincident points
                        if any(abs(v[0] - x) < 1e-6 and abs(v[1] - y) < 1e-6 for v in verticies):
                            continue

                        verticies.append((x, y, z_ft * 0.3048))

                    for i in reversed(range(len(verticies))):
                        # i is the "middle point"
                        x1 = verticies[(i-1) % len(verticies)][0]
                        y1 = verticies[(i-1) % len(verticies)][1]
                        x2 = verticies[i][0]
                        y2 = verticies[i][1]
                        x3 = verticies[(i+1) % len(verticies)][0]
                        y3 = verticies[(i+1) % len(verticies)][1]

                        # x1 y2 - x1 y3 + x2 y3 - x3 y2 + x3 y1 - x2 y1 = 0
                        if abs(x1 * y2 - x1 * y3 + x2 * y3 - x3 * y2 + x3 * y1 - x2 * y1) < 1e-6:
                            # Remove the middle point, or last current point on the list.
                            verticies.pop(i)

                    if lower_level_zone.name not in ceiling_counts:
                        ceiling_counts[lower_level_zone.name] = 1
                    else:
                        ceiling_counts[lower_level_zone.name] += 1
                    ceiling_n = ceiling_counts[lower_level_zone.name]

                    if upper_level_zone.name not in floor_counts:
                        floor_counts[upper_level_zone.name] = 1
                    else:
                        floor_counts[upper_level_zone.name] += 1
                    floor_n = floor_counts[upper_level_zone.name]

                    ceiling_name = f"{lower_level_zone.name_for_ref()} Ceiling {ceiling_n}"
                    floor_name = f"{upper_level_zone.name_for_ref()} Floor {floor_n}"

                    ceiling = IdfBuildingSurface()
                    ceiling.name(ceiling_name).surface_type("Ceiling").construction_name(construction_name).zone_name(lower_level_zone.name).outside_boundary_condition("Surface").outside_boundary_condition_object(floor_name).space_name(lower_level_zone.space_name)
                    assert signed_area(verticies) > 0
                    ceiling.vertices(verticies)

                    floor = IdfBuildingSurface()
                    floor.name(floor_name).surface_type("Floor").construction_name(construction_name).zone_name(upper_level_zone.name).outside_boundary_condition("Surface").outside_boundary_condition_object(ceiling_name).space_name(upper_level_zone.space_name)

                    floor_vertices = list(reversed(verticies))
                    assert signed_area(floor_vertices) < 0
                    floor.vertices(floor_vertices)

                    idf_surfaces.append(ceiling)
                    idf_surfaces.append(floor)


        # Done looping over all other zones, what is remaining is a roof.
        if lower_level_poly.area > 0:
            if isinstance(lower_level_poly, shapely.geometry.MultiPolygon):
                polygons = lower_level_poly.geoms
            else:
                polygons = [lower_level_poly]

            for poly in polygons:
                # Force Counter-clockwise
                poly = shapely.geometry.polygon.orient(poly, sign=1.0)
                verticies = []
                for point in poly.exterior.coords:
                    # Covert to m
                    x = point[0] * 0.3048
                    y = point[1] * 0.3048
                    # Check for coincident points
                    if any(abs(v[0] - x) < 1e-6 and abs(v[1] - y) < 1e-6 for v in verticies):
                        continue

                    verticies.append((x, y, z_ft * 0.3048))

                for i in reversed(range(len(verticies))):
                    # i is the "middle point"
                    x1 = verticies[(i-1) % len(verticies)][0]
                    y1 = verticies[(i-1) % len(verticies)][1]
                    x2 = verticies[i][0]
                    y2 = verticies[i][1]
                    x3 = verticies[(i+1) % len(verticies)][0]
                    y3 = verticies[(i+1) % len(verticies)][1]

                    # x1 y2 - x1 y3 + x2 y3 - x3 y2 + x3 y1 - x2 y1 = 0
                    if abs(x1 * y2 - x1 * y3 + x2 * y3 - x3 * y2 + x3 * y1 - x2 * y1) < 1e-6:
                        # Remove the middle point, or last current point on the list.
                        verticies.pop(i)

                if lower_level_zone.name not in ceiling_counts:
                    ceiling_counts[lower_level_zone.name] = 1
                else:
                    ceiling_counts[lower_level_zone.name] += 1
                ceiling_n = ceiling_counts[lower_level_zone.name]

                two_d_verticies = [(v[0], v[1]) for v in verticies]

                if not is_convex(two_d_verticies):
                    triangles = earclip(two_d_verticies)
                    triangle_num = 1
                    for triangle in triangles:
                        ceiling = IdfBuildingSurface()

                        ceiling.name(f"{lower_level_zone.name_for_ref()} Ceiling {ceiling_n}.{triangle_num}").surface_type("Roof")
                        ceiling.construction_name(roof_construction).zone_name(lower_level_zone.name).outside_boundary_condition("Outdoors")
                        ceiling.space_name(lower_level_zone.space_name)

                        ceiling_vertices = [(v[0], v[1], z_ft * 0.3048) for v in triangle]
                        assert signed_area(ceiling_vertices) > 0
                        ceiling.vertices(ceiling_vertices)
                        idf_surfaces.append(ceiling)
                        triangle_num += 1
                else:
                    ceiling = IdfBuildingSurface()
                    ceiling.name(f"{lower_level_zone.name_for_ref()} Ceiling {ceiling_n}").surface_type("Roof")
                    ceiling.construction_name(roof_construction).zone_name(lower_level_zone.name).outside_boundary_condition("Outdoors")
                    ceiling.space_name(lower_level_zone.space_name)
                    assert signed_area(verticies) > 0
                    ceiling.vertices(verticies)

                    idf_surfaces.append(ceiling)

    return idf_surfaces

class WallResult:
    """
    Contains pair of wall, overlap_wall, and overlap_ratio
    """
    def __init__(self, wall: NamedWall, overlap_wall: NamedWall | None, overlap_ratio: float) -> None:
        self.wall = wall
        self.overlap_wall = overlap_wall
        self.overlap_ratio = overlap_ratio

    def __str__(self) -> str:
        return f"Wall: {self.wall}, Overlap Wall: {self.overlap_wall}, Overlap Ratio: {self.overlap_ratio}"

    def __repr__(self) -> str:
        return self.__str__()

def analyze_wall_group_2(wall_group: list[NamedWall]) -> list[WallResult]:
    """
    wall_group is a list of NamedWall objects that are all on the same infinite line.
    """
    # Loop over walls.
    # Loop over other walls, calculate overlap for each.
    results: list[WallResult] = []

    # First assert that all walls have different names.
    names = set()
    dups = set()
    for wall in wall_group:
        if wall.name in names:
            dups.add(wall.name)
        names.add(wall.name)

    if len(dups) > 0:
        wall_names = ", ".join(sorted(dups))
        raise ValueError(f"Wall names must be unique. Duplicate names found: {wall_names}")

    tol = 1e-4

    all_walls = [w.clone() for w in wall_group]
    wall_stack = [w for w in all_walls]

    transformer = P_2d_to_1d(wall_group[0])

    while len(wall_stack) > 0:
        curr_wall = wall_stack.pop()

        for other_wall in all_walls:
            if curr_wall is other_wall:
                continue

            transformed11 = transformer.to_1d(curr_wall.wall.x1, curr_wall.wall.y1)
            transformed12 = transformer.to_1d(curr_wall.wall.x2, curr_wall.wall.y2)
            transformed21 = transformer.to_1d(other_wall.wall.x1, other_wall.wall.y1)
            transformed22 = transformer.to_1d(other_wall.wall.x2, other_wall.wall.y2)
            overlap_length, x1, x2 = overlap(transformed11, transformed12, transformed21, transformed22)

            if overlap_length < tol:
                continue

            # Else we need to break up our walls. At most, each wall can break into 3 parts.
            curr_wall_x1 = min(transformed11, transformed12)
            curr_wall_x2 = max(transformed11, transformed12)
            other_wall_x1 = min(transformed21, transformed22)
            other_wall_x2 = max(transformed21, transformed22)

            def curr_wall_trans(x1, x2):
                if transformed11 < transformed12:
                    p1 = transformer.to_2d(min(x1, x2))
                    p2 = transformer.to_2d(max(x1, x2))
                else:
                    p1 = transformer.to_2d(max(x1, x2))
                    p2 = transformer.to_2d(min(x1, x2))
                return Wall(p1[0], p1[1], p2[0], p2[1], curr_wall.wall.int_ext)

            def other_wall_trans(x1, x2):
                if transformed21 < transformed22:
                    p1 = transformer.to_2d(min(x1, x2))
                    p2 = transformer.to_2d(max(x1, x2))
                else:
                    p1 = transformer.to_2d(max(x1, x2))
                    p2 = transformer.to_2d(min(x1, x2))
                return Wall(p1[0], p1[1], p2[0], p2[1], other_wall.wall.int_ext)

            # Need to maintain the proper direction on these.

            # Update the Current Walls
            if curr_wall_x1 - x1 < -tol:
                w = NamedWall(curr_wall.zone, f"{curr_wall.name}.1", curr_wall_trans(curr_wall_x1, x1), curr_wall.space)
                wall_stack.append(w)
                all_walls.append(w)

            overlap_wall_1 = NamedWall(curr_wall.zone, f"{curr_wall.name}.2", curr_wall_trans(x1, x2), curr_wall.space)

            if curr_wall_x2 - x2 > tol:
                w = NamedWall(curr_wall.zone, f"{curr_wall.name}.3", curr_wall_trans(x2, curr_wall_x2), curr_wall.space)
                wall_stack.append(w)
                all_walls.append(w)

            # Update the Other Walls
            if other_wall_x1 - x1 < -tol:
                w = NamedWall(other_wall.zone, f"{other_wall.name}.1", other_wall_trans(other_wall_x1, x1), other_wall.space)
                wall_stack.append(w)
                all_walls.append(w)

            overlap_wall_2 = NamedWall(other_wall.zone, f"{other_wall.name}.2", other_wall_trans(x1, x2), other_wall.space)

            if other_wall_x2 - x2 > tol:
                w = NamedWall(other_wall.zone, f"{other_wall.name}.3", other_wall_trans(x2, other_wall_x2), other_wall.space)
                wall_stack.append(w)
                all_walls.append(w)

            # Remove the current wall and add the new walls
            all_walls.remove(curr_wall)
            all_walls.remove(other_wall)
            wall_stack.remove(other_wall)

            # Check that the walls are not assigned to the same zone/space:
            # This allows us to pass in several "zone" objects all tied back to the same zone name, and have the overlap sections magically erased.
            if not ((curr_wall.space.strip() == '' and other_wall.space.strip() == '' and  curr_wall.zone == other_wall.zone) or (curr_wall.space.strip() != '' and other_wall.space.strip() != '' and curr_wall.space == other_wall.space)):
                results.append(WallResult(overlap_wall_1, overlap_wall_2, 1))
                results.append(WallResult(overlap_wall_2, overlap_wall_1, 1))

            break  # Break out of the inner loop
        # If we went through every other wall possible and found no overlaps, then this is an outdoor wall.
        else:
            results.append(WallResult(curr_wall, None, 0))

    return results


def wall_analyze(wall_groups: list[list[NamedWall]]) -> list[WallResult]:
    """
    wall_groups is a list of lists of NamedWall objects. 
    Each inner list is a group of walls that are on the same line in the 2-D plane.
    """
    all_matches: list[WallResult] = []
    for group in wall_groups:
        matches: list[WallResult] = analyze_wall_group_2(group)
        all_matches.extend(matches)
    return all_matches

class Rect:
    """x1, y1 is the bottom left corner of the rectangle"""
    def __init__(self, x1, y1, w, h) -> None:
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h

        self.bottom = y1
        self.top = y1 + h
        self.left = x1
        self.right = x1 + w

    def points(self):
        return [P(self.x1, self.y1), P(self.x1 + self.w, self.y1), P(self.x1 + self.w, self.y1 + self.h), P(self.x1, self.y1 + self.h)]

    def walls(self, int_ext):
        """int_ext is a list of length 4, each element is either 'i' or 'e'. Order is bottom, right, top, left."""
        # check that length of int_ext equals 4
        if len(int_ext) != 4:
            raise ValueError('Length of int_ext must equal 4')

        points = self.points()
        return p2w(points, int_ext)

    def down(self, dy):
        return Rect(self.x1, self.y1 - dy, self.w, self.h)

    def up(self, dy):
        return Rect(self.x1, self.y1 + dy, self.w, self.h)

    def left(self, dx):
        return Rect(self.x1 - dx, self.y1, self.w, self.h)

    def right(self, dx):
        return Rect(self.x1 + dx, self.y1, self.w, self.h)

    def top_left(self):
        return P(self.x1, self.y1 + self.h)

    def top_right(self):
        return P(self.x1 + self.w, self.y1 + self.h)

    def bottom_left(self):
        return P(self.x1, self.y1)

    def bottom_right(self):
        return P(self.x1 + self.w, self.y1)

    def top_edge(self):
        return self.y1 + self.h

    def bottom_edge(self):
        return self.y1

    def left_edge(self):
        return self.x1

    def right_edge(self):
        return self.x1 + self.w

class P:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def right(self, dx):
        return P(self.x + dx, self.y)

    def left(self, dx):
        return P(self.x - dx, self.y)

    def up(self, dy):
        return P(self.x, self.y + dy)

    def down(self, dy):
        return P(self.x, self.y - dy)

    def set_x(self, x):
        return P(x, self.y)

    def set_y(self, y):
        return P(self.x, y)

    # Override equality operator
    def __eq__(self, other):
        tol = 1e-6
        return abs(self.x - other.x) < tol and abs(self.y - other.y) < tol

    def to_1d(self):
        return p_2d_to_1d(self)

    def clone(self):
        return P(self.x, self.y)

def p2w(points: list[P], int_ext: list[str]) -> list[Wall]:
    walls = []
    # Check that length of int_ext equals length of points
    if len(points) != len(int_ext):
        raise ValueError('Length of int_ext must equal length of points')

    for i in range(len(points)):
        if i == len(points) - 1:
            walls.append(Wall(points[i].x, points[i].y, points[0].x, points[0].y, int_ext[i]))
        else:
            walls.append(Wall(points[i].x, points[i].y, points[i + 1].x, points[i + 1].y, int_ext[i]))

    return walls


def print_idf_wall(wall_match: WallResult, height_ft, z_ft, exterior_wall_construction, interior_wall_construction = "Interior Wall") -> IdfBuildingSurface:
    """wall match is a tuple of (wall, overlap_wall, overlap_ratio)"""
    wall = wall_match.wall
    overlap_wall = wall_match.overlap_wall
    overlap_ratio = wall_match.overlap_ratio

    zone_name = wall.zone
    space_name = wall.space

    surface = IdfBuildingSurface()
    surface.name(wall.name).surface_type("Wall").zone_name(zone_name).space_name(space_name)

    if overlap_wall is None:
        surface.outside_boundary_condition("Outdoors").outside_boundary_condition_object("")
        surface.sun_exposure("SunExposed").wind_exposure("WindExposed").construction_name(exterior_wall_construction)
    else:
        surface.outside_boundary_condition("Surface").outside_boundary_condition_object(overlap_wall.name)
        surface.sun_exposure("NoSun").wind_exposure("NoWind").construction_name(interior_wall_construction)

    vertices = []
    x1 = wall.wall.x1 * 0.3048
    y1 = wall.wall.y1 * 0.3048
    x2 = wall.wall.x2 * 0.3048
    y2 = wall.wall.y2 * 0.3048
    z = z_ft * 0.3048
    height = height_ft * 0.3048
    vertices.append((x1, y1, z))
    vertices.append((x2, y2, z))
    vertices.append((x2, y2, z + height))
    vertices.append((x1, y1, z + height))
    surface.vertices(vertices)

    return surface

class IdfZone:
    def __init__(self) -> None:
        self._name = ""
        self._origin_x_ft = 0
        self._origin_y_ft = 0
        self._origin_z_ft = 0

    def name(self, name):
        self._name = name
        return self

    def origin_x_ft(self, origin_x_ft):
        self._origin_x_ft = origin_x_ft
        return self

    def origin_y_ft(self, origin_y_ft):
        self._origin_y_ft = origin_y_ft
        return self

    def origin_z_ft(self, origin_z_ft):
        self._origin_z_ft = origin_z_ft
        return self

    def __str__(self) -> str:

        origin_x_m = self._origin_x_ft * 0.3048
        origin_y_m = self._origin_y_ft * 0.3048
        origin_z_m = self._origin_z_ft * 0.3048

        lines = []
        lines.append(f"Zone,")
        lines.append(f"  {self._name},")
        lines.append(f"  0, !- Direction of Relative North")
        lines.append(f"  {origin_x_m}, !- {self.origin_x_ft} ft - X Origin")
        lines.append(f"  {origin_y_m}, !- {self.origin_y_ft} ft - Y Origin")
        lines.append(f"  {origin_z_m}, !- {self.origin_z_ft} ft - Z Origin")


        return "".join([line + "\n" for line in lines])


def zones_from_file(filepath) -> list[tuple[Z, int]]:
    """
    Return tuple of the zone and the floor that it's on.
    """
    with open(filepath, encoding="utf-8") as file:
        data = [line.split("\t") for line in file.read().splitlines()]

    zones: list[tuple[Z, int]] = []

    for row in data:
        if len(row) < 8:
            raise ValueError("Invalid data format")

        name = row[0]
        floor = int(row[1])
        points = []
        for i in range(2, len(row), 2):
            x = float(row[i])
            y = float(row[i+1])
            points.append(P(x, y))

        walls = p2w(points, ['i'] * len(points))

        if name == 'A124':
            zones.append((Z(name, walls, 0, 0, 0, ''), floor))
        else:
            zone_name = space_zone_map[name]
            zones.append((Z(zone_name, walls, 0, 0, 0, name), floor))

    return zones

def all_wall_data_from_zones(zones: list[Z], add_text) -> list[NamedWall]:
    zone_name_groups: dict[str, list[Z]] = {}
    for z in zones:
        if z.name not in zone_name_groups:
            zone_name_groups[z.name] = []
        zone_name_groups[z.name].append(z)

    all_wall_data: list[NamedWall] = []
    for _, zone_group in zone_name_groups.items():
        # This is so all walls are unique, even if we have two 'zones' with the same name.
        start_num = 1
        for z in zone_group:
            wall_data = z.wall_data(start_num, add_text)
            all_wall_data.extend(wall_data)
            start_num += len(wall_data)

    return all_wall_data

output_dict = {}
zones = []

def order_walls_counter_clockwise(walls: list[Wall]) -> list[Wall]:
    walls_to_sort = [w for w in walls[1:]]

    new_walls: list[Wall] = []
    new_walls.append(walls[0])

    tol = 1e-6

    while len(walls_to_sort) > 0:
        i = 0
        found = False

        while i < len(walls_to_sort):
            xdiff = abs(walls_to_sort[i].x1 - new_walls[-1].x2)
            ydiff = abs(walls_to_sort[i].y1 - new_walls[-1].y2)

            if xdiff < tol and ydiff < tol: 
                new_walls.append(walls_to_sort[i])
                walls_to_sort.pop(i)
                found = True
                break
            i += 1

        if not found:
            raise ValueError(f"Found unconnected walls {i} {len(walls_to_sort)}.\nNew walls ({len(new_walls)})\n{new_walls}\nWalls to sort ({len(walls_to_sort)})\n{walls_to_sort}")

    # Check that the signed area is positive
    if signed_area_from_walls(new_walls) < 0:
        new_walls.reverse()

    # Collinear points is removed later
    return new_walls


def test_wall_ordering():
    wall_1 = Wall(0, 0, 1, 0, 'i')
    wall_2 = Wall(1, 0, 1, 1, 'i')
    wall_3 = Wall(1, 1, 0, 1, 'i')
    wall_4 = Wall(0, 1, 0, 0, 'i')

    # Try mixed up.
    walls = [wall_3, wall_2, wall_1, wall_4]

    ordered_wall_1, ordered_wall_2, ordered_wall_3, ordered_wall_4 = order_walls_counter_clockwise(walls)

    assert ordered_wall_1 == wall_3
    assert ordered_wall_2 == wall_4
    assert ordered_wall_3 == wall_1
    assert ordered_wall_4 == wall_2


def gen_windows(window_data: WindowData, win_height_ft = 6.0, z_ft = 4.0):
    xs = spacing_around(window_data.surface_length_ft, window_data.width_ft, window_data.count)

    window_surfaces = []

    for i, x in enumerate(xs, 1):
        window_surface = WindowSurface()
        window_surface.name(f"{window_data.surface} Window {i}")
        window_surface.building_surface_name(window_data.surface)
        window_surface.construction_name("ASHRAE 90.1-2010 CZ4 Metal Framing Fenestration")
        window_surface.height(win_height_ft * 0.3048)
        window_surface.length(window_data.width_ft * 0.3048)
        window_surface.starting_x(x * 0.3048)
        window_surface.starting_z(z_ft * 0.3048)

        window_surfaces.append(window_surface)

    return window_surfaces


def wall_test_with_spaces():
    rect1 = Rect(0, 0, 10, 10)
    rect2 = Rect(10, 5, 10, 10)

    space1 = Z("Zone 1", rect1.walls(['i', 'i', 'i', 'i']), 0, 0, 0, "Space 1")
    space2 = Z("Zone 1", rect2.walls(['i', 'i', 'i', 'i']), 0, 0, 0, "Space 2")

    grouped_walls = group_walls(all_wall_data_from_zones([space1, space2], ''))
    matches = wall_analyze(grouped_walls)

    for match in matches:
        wall_surface = print_idf_wall(match, 10, 0, "Exterior Wall")
        print(str(wall_surface))
