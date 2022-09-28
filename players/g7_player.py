import functools
import heapq
import logging
# import os
# import pickle
from typing import Iterator, Tuple, Union  # List

from matplotlib.path import Path
import numpy as np
# import scipy
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon as ShapelyPolygon
import sympy
from sympy import Triangle, atan2
from sympy.geometry import Point2D, Polygon
import constants
import math

STEP = 5.0  # chunk size
DIST = scipy_stats.norm(0, 1)


@functools.lru_cache()
def standard_ppf(conf: float) -> float:
    return DIST.ppf(conf)

def dist2(p1: Point, p2: Point) -> float:
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def dist1(p1: Point, p2: Point = Point(0, 0)) -> float:
        return math.sqrt(dist2(p1, p2))

def polygon_to_points(golf_map: sympy.Polygon) -> Iterator[Tuple[float, float]]:
    """
    This function takes in the polygon golf map and returns an iterator for the
    points on a lattice with distance STEP. We ignore the edges of the map
    where there is only water.
    """
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    for point in golf_map.vertices:
        x, y = float(point.x), float(point.y)
        x_min = min(x, x_min)
        x_max = max(x, x_max)
        y_min = min(y, y_min)
        y_max = max(y, y_max)

    x_step = STEP
    y_step = STEP

    x_current, y_current = x_min, y_min
    while x_current < x_max:
        while y_current < y_max:
            yield float(x_current), float(y_current)
            y_current += y_step
        y_current = y_min
        x_current += x_step


def sympy_polygon_to_shapely(polygon: sympy.Polygon) -> ShapelyPolygon:
    """
    Helper function that converts a sympy polygon to a shapely polygon
    """
    vertices = polygon.vertices
    vertices.append(vertices[0])
    return ShapelyPolygon(vertices)


def sympy_tri_to_shapely(sympy_tri: Triangle) -> ShapelyPolygon:
    """Helper function to convert sympy Polygon to shapely Polygon object"""
    v = sympy_tri.vertices
    vert_list = []
    for i in v:
        vert_list.append(Point2D(i[0], i[1]))
    vert_list.append(Point2D(v[0][0], v[0][1]))
    return ShapelyPolygon(vert_list)


def sympy_polygon_to_mpl(sympy_poly: Polygon) -> Path:
    """Helper function to convert sympy Polygon to matplotlib Path object"""
    v = sympy_poly.vertices
    v = list(v)
    v.append(v[0])
    return Path(v, closed=True)


def sympy_tri_to_mpl(sympy_tri: Triangle) -> Path:
    """Helper function to convert sympy Polygon to matplotlib Path object"""
    v = sympy_tri.vertices
    vert_list = []
    for i in v:
        vert_list.append(Point2D(i[0], i[1]))
    vert_list.append(Point2D(v[0][0], v[0][1]))
    return Path(vert_list, closed=True)


def spread_points(current_point,
                  angles: np.array,
                  distance,
                  reverse) -> np.array:
    curr_x, curr_y = current_point
    if reverse:
        angles = np.flip(angles)
    xs = np.cos(angles) * distance + curr_x
    ys = np.sin(angles) * distance + curr_y
    return np.column_stack((xs, ys))


def splash_zone(distance: float,
                angle: float,
                conf: float,
                skill: int,
                current_point: Tuple[float, float]) -> np.array:
    conf_points = np.linspace(1 - conf, conf, 5)
    distances = np.vectorize(standard_ppf)(conf_points) * (distance / skill) + distance
    angles = np.vectorize(standard_ppf)(conf_points) * (1/(2*skill)) + angle
    scale = 1.1 if distance <= 20 else 1.0

    max_distance = distances[-1]*scale
    top_arc = spread_points(current_point, angles, max_distance, False)

    if distance > 20:
        min_distance = distances[0]
        bottom_arc = spread_points(current_point, angles, min_distance, True)
        return np.concatenate((top_arc, bottom_arc, np.array([top_arc[0]])))

    current_point = np.array([current_point])
    return np.concatenate((current_point, top_arc, current_point))


class ScoredPoint:
    """Scored point class for use in A* search algorithm
    Building off of last years g2 player
    """
    def __init__(self,
                 point: Tuple[float, float],
                 goal: Tuple[float, float],
                 actual_cost=float('inf'),
                 previous=None,
                 goal_dist=None,
                 skill=50) -> None:

        self.point = point
        self.goal = goal

        self.previous = previous

        self._actual_cost = actual_cost
        if goal_dist is None:
            a = np.array(self.point)
            b = np.array(self.goal)
            goal_dist = np.linalg.norm(a - b)

        max_target_dist = 200 + skill
        max_dist = standard_ppf(0.99) * (max_target_dist / skill) + max_target_dist
        max_dist *= 1.10

        self._h_cost = goal_dist / max_dist

        self._f_cost = self.actual_cost + self.h_cost

    @property
    def f_cost(self):
        return self._f_cost

    @property
    def h_cost(self):
        return self._h_cost

    @property
    def actual_cost(self):
        return self._actual_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.point == other.point

    def __hash__(self):
        return hash(self.point)

    def __repr__(self):
        return f"ScoredPoint(point = {self.point}, h_cost = {self.h_cost})"


class Player:
    def __init__(self,
                 skill: int,
                 rng: np.random.Generator,
                 logger: logging.Logger,
                 golf_map: sympy.Polygon,
                 start: Point2D,
                 target: Point2D,
                 sand_traps: list[sympy.Polygon],
                 map_path: str,
                 precomp_dir: str) -> None:
        """Initialise the player with given skill.
        Args:
        skill (int): skill of your player
        rng (np.random.Generator): np random number generator, use for same player behavior
        logger (logging.Logger): logger use this like logger.info("message")
        golf_map (sympy.Polygon): Golf Map polygon
        start (Point2D): Start location
        target (Point2D): Target location
        map_path (str): File path to map
        precomp_dir (str): Directory path to store/load precomputation
        """

        self.skill = skill
        self.conf = 0.95
        self.rng = rng
        self.logger = logger
        self.np_points = None
        self.mpl_poly = None
        self.shapely_poly = None
        self.mpl_poly_trap = [sympy_polygon_to_mpl(sandtrap) for sandtrap in sand_traps]
        self.shapely_poly_trap = None
        self.goal = None
        self.poly_list = []
        self.poly_shapely = []
        self.prev_rv = None

        self.max_distance = 200 + self.skill
        self.max_ddist = scipy_stats.norm(self.max_distance, self.max_distance / self.skill)
        self.max_ddist_sand = scipy_stats.norm(self.max_distance / 2, 2 * self.max_distance / self.skill)

# # Group 9 code needed for precompute() ################################
# self.rows, self.columns = None, None
# self.dmap, self.pmap = None, None
# self.quick_map = ShapelyPolygon([(p.x, p.y) for p in golf_map.vertices])
# self.quick_sand = [ShapelyPolygon([(p.x, p.y) for p in sand_trap.vertices]) \
#                    for sand_trap in sand_traps]

# x_min, y_min, max_x, max_y = self.quick_map.bounds
# self.min_x = x_min
# self.min_y = y_min
# width, height = max_x - x_min, max_y - y_min

# self.rows = int(np.ceil(height / STEP))
# self.columns = int(np.ceil(width / STEP )) # STEP == self.cell_width??
# self.zero_center = Point(x_min + STEP / 2, max_y - STEP / 2)

#    precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))
#     # precompute check
#     if os.path.isfile(precomp_path):
#         self.dmap = pickle.load(open(precomp_path, "rb"))
#     else:
#         self.precompute()
#         pickle.dump(self.dmap, open(precomp_path, "wb"))
# End #################################################################

    @functools.lru_cache()
    def _max_ddist_ppf(self, conf: float):
        return self.max_ddist.ppf(1.0 - conf)

    @functools.lru_cache()
    def _max_ddist_sand_ppf(self, conf: float):
        return self.max_ddist_sand.ppf(1.0 - conf)

    def numpy_adjacent_and_dist(self, point: Tuple[float, float], conf: float):
        is_in_sandtrap = any([sandtrap.contains_point(point) for sandtrap in self.mpl_poly_trap])
        cloc_distances = cdist(self.np_points, np.array([np.array(point)]), 'euclidean')
        cloc_distances = cloc_distances.flatten()

        if is_in_sandtrap:
            distance_mask = cloc_distances <= self._max_ddist_sand_ppf(conf)
        else:
            distance_mask = cloc_distances <= self._max_ddist_ppf(conf)

        reachable_points = self.np_points[distance_mask]
        goal_distances = self.np_goal_dist[distance_mask]

        return reachable_points, goal_distances

    def next_target(self,
                    curr_loc: Tuple[float, float],
                    goal: Point2D,
                    conf: float) -> Union[None, Tuple[float, float]]:
        point_goal = float(goal.x), float(goal.y)
        heap = [ScoredPoint(curr_loc, point_goal, 0.0)]
        start_point = heap[0].point
        # Cache the best cost and avoid adding useless points to the heap
        best_cost = {tuple(curr_loc): 0.0}
        visited = set()
        points_checked = 0
        while len(heap) > 0:
            next_sp = heapq.heappop(heap)
            next_p = next_sp.point

            if next_p in visited:
                continue
            if next_sp.actual_cost > 10:
                continue
            if next_sp.actual_cost > 0 and not self.is_splash_zone_within_polygon(next_sp.previous.point, next_p, conf):
                if next_p in best_cost:
                    del best_cost[next_p]
                continue
            visited.add(next_p)

            if np.linalg.norm(np.array(self.goal) - np.array(next_p)) <= 5.4 / 100.0:
                # All we care about is the next point
                # TODO: We need to check if the path length is <= 10,
                # because if it isn't we probably need to reduce the conf and
                # try again for a shorter path.
                while next_sp.previous.point != start_point:
                    next_sp = next_sp.previous
                return next_sp.point

            # Add adjacent points to heap
            reachable_points, goal_dists = self.numpy_adjacent_and_dist(next_p, conf)
            for i in range(len(reachable_points)):
                candidate_point = tuple(reachable_points[i])
                goal_dist = goal_dists[i]
                new_point = ScoredPoint(candidate_point, point_goal, next_sp.actual_cost + 1, next_sp,
                                        goal_dist=goal_dist, skill=self.skill)
                if candidate_point not in best_cost or best_cost[candidate_point] > new_point.actual_cost:
                    points_checked += 1
                    # if not self.splash_zone_within_polygon(new_point.previous.point, new_point.point, conf):
                    #     continue
                    best_cost[new_point.point] = new_point.actual_cost
                    heapq.heappush(heap, new_point)

        # No path available
        return None

    def polygon_to_np_points(self,
                             goal: Tuple[float, float],
                             golf_map: Polygon,
                             sand_traps: list[Polygon]):
        # Storing the points as numpy array
        np_points = [goal]
        map_points = [goal]
        trap_points = []
        self.mpl_poly = sympy_polygon_to_mpl(golf_map)
        self.shapely_poly = sympy_polygon_to_shapely(golf_map)
        pp = list(polygon_to_points(golf_map))
        for point in pp:
            # no = True
            # Use matplotlib here because it's faster than shapely for this calculation...
            # for trap in sand_traps:
            #     self.mpl_poly_trap = sympy_tri_to_mpl(trap)
            #     if self.mpl_poly_trap.contains_point(point):
            #         no = False
            if self.mpl_poly.contains_point(point):  # and no:
                # map_points.append(point)
                x, y = point
                np_points.append(np.array([x, y]))
        # self.map_poinats = np.array(map_points)
        self.np_points = np.array(np_points)
        self.np_goal_dist = cdist(self.np_points, np.array([np.array(self.goal)]), 'euclidean')
        self.np_goal_dist = self.np_goal_dist.flatten()

    def reachable(self,
                  current_point: Tuple[float, float],
                  target_point: Tuple[float, float],
                  conf: float) -> bool:
        if type(current_point) == Point2D:
            current_point = tuple(current_point)
        if type(target_point) == Point2D:
            target_point = tuple(target_point)

        current_point = np.array(current_point).astype(float)
        target_point = np.array(target_point).astype(float)

        return np.linalg.norm(current_point - target_point) <= self._max_ddist_ppf(conf)

    def is_splash_zone_within_polygon(self,
                                      current_point: Tuple[float, float],
                                      target_point: Tuple[float, float],
                                      conf: float) -> bool:
        if type(current_point) == Point2D:
            current_point = tuple(Point2D)
        if type(target_point) == Point2D:
            target_point = tuple(Point2D)

        is_in_sandtrap = any([sandtrap.contains_point(target_point) for sandtrap in self.mpl_poly_trap])
        if is_in_sandtrap:
            distance = np.linalg.norm(np.array(current_point).astype(float) - np.array(target_point).astype(float))
        if self.skill < 50:
            distance = (np.linalg.norm(np.array(current_point).astype(float) - np.array(target_point).astype(float)) * 1.05)
        elif self.skill >= 50:
            distance = (np.linalg.norm(np.array(current_point).astype(float) - np.array(target_point).astype(float)) * 1.08)
        cx, cy = float(current_point[0]), float(current_point[1])
        tx, ty = float(target_point[0]), float(target_point[1])
        angle = np.arctan2(ty - cy, tx - cx)
        # for trap in self.poly_shapely:
        #     if trap.contains(ShapelyPolygon(splash_zone_poly_points)):
        #         return False
        splash_zone_polygon_points = splash_zone(float(distance), float(angle), float(conf), self.skill, current_point)
        return self.shapely_poly.contains(ShapelyPolygon(splash_zone_polygon_points))

    def play(self,
             score: int,
             golf_map: sympy.Polygon,
             target: Point2D,
             sand_traps: list[sympy.Polygon],
             curr_loc: Point2D,
             prev_loc: Point2D,
             prev_landing_point: Point2D,
             prev_admissible: bool) -> Tuple[float, float]:
        """Function which based on current game state returns the distance and angle, the shot must be played
        Args:
        score (int): Your total score including current turn
        golf_map (sympy.Polygon): Golf Map polygon
        target (Point2D): Target location
        curr_loc (Point2D): Your current location
        prev_loc (Point2D): Your previous location. If you haven't played previously then None
        prev_landing_point (Point2D): Your previous shot landing location. If you haven't played previously then None
        prev_admissible (bool): Boolean stating if your previous shot was within the polygon limits. If you haven't played previously then None
        Returns:
        Tuple[float, float]: Return a tuple of distance and angle in radians to play the shot
        """
        if self.np_points is None:
            gx, gy = float(target.x), float(target.y)
            self.goal = float(target.x), float(target.y)
            self.polygon_to_np_points((gx, gy), golf_map, sand_traps)

        # Optimization to retry missed shots
        if self.prev_rv is not None and curr_loc == prev_loc:
            return self.prev_rv

        target_point = None
        confidence = self.conf
        cl = float(curr_loc.x), float(curr_loc.y)
        while target_point is None:
            if confidence <= 0.5:
                return None

            # print(f"searching with {confidence} confidence")
            target_point = self.next_target(cl, target, confidence)
            confidence -= 0.05

        # fixup target
        current_point = np.array(tuple(curr_loc)).astype(float)
        if tuple(target_point) == self.goal:
            original_dist = np.linalg.norm(np.array(target_point) - current_point)
            v = np.array(target_point) - current_point
            # Unit vector pointing from current to target
            u = v / original_dist
            if np.linalg.norm(current_point - self.goal) <= 20:
                cx, cy = current_point
                tx, ty = target_point
                angle = np.arctan2(ty - cy, tx - cx)
                rv = min(((dist1(target_point, current_point) / (1 - (1 / self.skill * 3)))+3), constants.min_putter_dist), angle
                self.prev_rv = rv
                return rv
            roll_distance = original_dist / 20
            max_offset = roll_distance
            offset = 0
            prev_target = target_point
            while offset < max_offset and self.is_splash_zone_within_polygon(tuple(current_point), target_point, confidence):
                offset += 1
                dist = original_dist - offset
                prev_target = target_point
                target_point = current_point + u * dist
            target_point = prev_target
            
        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(ty - cy, tx - cx)

        rv = curr_loc.distance(Point2D(target_point, evaluate=False)), angle
        self.prev_rv = rv
        return rv
