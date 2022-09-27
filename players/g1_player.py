import os
import pickle
import numpy as np
import functools
import sympy
import logging
import heapq
from scipy import stats as scipy_stats

from typing import Tuple, Iterator, List, Union, Set
from sympy.geometry import Polygon, Point2D
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from scipy.spatial.distance import cdist

import constants


# Cached distribution
DIST = scipy_stats.norm(0, 1)
X_STEP = 5.0
Y_STEP = 5.0


@functools.lru_cache()
def standard_ppf(conf: float) -> float:
    return DIST.ppf(conf)


def result_point(distance: float, angle: float, current_point: Tuple[float, float]) -> Tuple[float, float]:
    cx, cy = current_point
    nx = cx + distance * np.cos(angle)
    ny = cy + distance * np.sin(angle)
    return nx, ny


def spread_points(current_point, angles: np.array, distance, reverse) -> np.array:
    curr_x, curr_y = current_point
    if reverse:
        angles = np.flip(angles)
    xs = np.cos(angles) * distance + curr_x
    ys = np.sin(angles) * distance + curr_y
    return np.column_stack((xs, ys))


def splash_zone(distance: float, angle: float, conf: float, skill: int, current_point: Tuple[float, float], in_sandtrap: bool, target_in_sand: bool) -> np.array:
    conf_points = np.linspace(1 - conf, conf, 5)
    if in_sandtrap:
        # the distance rating is halved, and the standard deviations for the angle and distance distributions are doubled
        distances = np.vectorize(standard_ppf)(
            conf_points) * (distance / skill) * 2 + distance / 2
        angles = np.vectorize(standard_ppf)(
            conf_points) * (1/(2*skill)) * 2 + angle
    else:
        distances = np.vectorize(standard_ppf)(
            conf_points) * (distance / skill) + distance
        angles = np.vectorize(standard_ppf)(
            conf_points) * (1/(2*skill)) + angle

    scale = 1.1
    putter_shot = distance < 20 and not in_sandtrap

    # A ball will not roll in two cases:
    #   1. player makes a putter shot
    # . 2. ball lands in a sand trap
    if putter_shot or target_in_sand:
        scale = 1.0

    max_distance = distances[-1]*scale
    top_arc = spread_points(current_point, angles, max_distance, False)

    if putter_shot:
        # return a fan region, instead of splash zone, this is so that we can
        # check whether the putter shot would pass by water 
        current_point = np.array([current_point])
        return np.concatenate((current_point, top_arc, current_point))
    else:
        min_distance = distances[0]
        bottom_arc = spread_points(current_point, angles, min_distance, True)
        return np.concatenate((top_arc, bottom_arc, np.array([top_arc[0]])))


def poly_to_points(poly: Polygon) -> Iterator[Tuple[float, float]]:
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for point in poly.vertices:
        x = float(point.x)
        y = float(point.y)
        x_min = min(x, x_min)
        x_max = max(x, x_max)
        y_min = min(y, y_min)
        y_max = max(y, y_max)
    x_step = X_STEP
    y_step = Y_STEP

    x_current = x_min
    y_current = y_min
    while x_current < x_max:
        while y_current < y_max:
            yield float(x_current), float(y_current)
            y_current += y_step
        y_current = y_min
        x_current += x_step


def sympy_poly_to_mpl(sympy_poly: Polygon) -> Path:
    """Helper function to convert sympy Polygon to matplotlib Path object"""
    # A polygon of 3 points will be typed as a Triangle and the its vertices is a tuple
    # so we need to transform it into list here for append to work
    v = list(sympy_poly.vertices)
    v.append(v[0])
    return Path(v, closed=True)


def sympy_poly_to_shapely(sympy_poly: Polygon) -> ShapelyPolygon:
    """Helper function to convert sympy Polygon to shapely Polygon object"""
    v = list(sympy_poly.vertices)
    v.append(v[0])
    return ShapelyPolygon(v)


def is_in_sand_trap(point: Tuple[float, float], sand_traps: List[Path], cache: Set[Tuple[float, float]] = set()) -> bool:
    """Returns True if @point is in the sand trap, otherwise False.

    @param cache: set of points known to be in sand trap, e.g. pre-computed map points that are in sand traps.
    """
    if point in cache:
        return True

    for sand_trap_poly in sand_traps:
        if sand_trap_poly.contains_point(point):
            return True

    return False


def find_map_points_in_sand_trap(map_points: List[Tuple[float, float]], sand_traps: List[Path]) -> Set[Tuple[float, float]]:
    """Returns a set of map points that are in any of the sand traps."""
    points_in_sand_trap = set()

    for map_point in map_points:
        if is_in_sand_trap(map_point, sand_traps):
            points_in_sand_trap.add(map_point)

    return points_in_sand_trap

def roll(current_point: Tuple[float, float], target_point: Tuple[float, float], rolling_factor: float) -> Tuple[float, float]:
    """Returns the final point along the line formed by @current_point, @target_point,
    but (1.0 + @rolling_factor) times the distance between @current_point and @target_point
    away from the @current_point.
    """
    cx, cy = current_point
    tx, ty = target_point
    curr_loc = Point2D(current_point)

    angle = np.arctan2(ty - cy, tx - cx)
    distance = curr_loc.distance(Point2D(target_point, evaluate=False))

    final_x, final_y = sympy.Point2D(
        curr_loc.x + (1. + rolling_factor) * distance * sympy.cos(angle),
        curr_loc.y + (1. + rolling_factor) * distance * sympy.sin(angle))
    

    return (float(final_x), float(final_y))

class ScoredPoint:
    """Scored point class for use in A* search algorithm"""

    def __init__(self, point: Tuple[float, float], goal: Tuple[float, float], actual_cost=float('inf'), previous=None, goal_dist=None, skill=50, sand_penalty=0):
        self.point = point
        self.goal = goal

        self.previous = previous

        self._actual_cost = actual_cost
        if goal_dist is None:
            a = np.array(self.point)
            b = np.array(self.goal)
            goal_dist = np.linalg.norm(a - b)

        max_target_dist = 200 + skill
        max_dist = standard_ppf(
            0.99) * (max_target_dist / skill) + max_target_dist
        max_dist *= 1.10
        self._h_cost = goal_dist / max_dist + sand_penalty

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
    AVOID_SAND_PENALTY = 0.1
    PUTTER_OVER_SAND_PENALTY = 1

    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger, golf_map: sympy.Polygon, start: sympy.geometry.Point2D, target: sympy.geometry.Point2D, sand_traps: List[sympy.Polygon], map_path: str, precomp_dir: str) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
            golf_map (sympy.Polygon): Golf Map polygon
            start (sympy.geometry.Point2D): Start location
            target (sympy.geometry.Point2D): Target location
            map_path (str): File path to map
            precomp_dir (str): Directory path to store/load precomputation
        """
        # # if depends on skill
        # precomp_path = os.path.join(precomp_dir, "{}_skill-{}.pkl".format(map_path, skill))
        # # if doesn't depend on skill
        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.np_map_points = None
        self.map_points_in_sand_trap = None  # :: Set[Tuple[float, float]]
        self.sand_trap_matlab_polys = None   # :: List[Path]
        self.sand_trap_shapely_polys = None
        self.mpl_paly = None
        self.shapely_poly = None
        self.goal = None
        self.prev_rv = None

        # Cached data
        max_dist = 200 + self.skill
        self.max_ddist = scipy_stats.norm(max_dist, max_dist / self.skill)

        max_sandtrap_dist = max_dist/2
        self.max_sandtrap_ddist = scipy_stats.norm(max_sandtrap_dist, max_dist / self.skill)

        # Conf level
        self.conf = 0.95
        if self.skill < 40:
            self.conf = 0.75

        # initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    @functools.lru_cache()
    def _max_ddist_ppf(self, conf: float):
        return self.max_ddist.ppf(1.0 - conf)

    def _max_sandtrap_ddist_ppf(self, conf:float):
        return self.max_sandtrap_ddist.ppf(1.0-conf)

    def splash_zone_within_polygon(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float) -> [bool, bool]:
        if type(current_point) == Point2D:
            current_point = tuple(Point2D)

        if type(target_point) == Point2D:
            target_point = tuple(Point2D)

        distance = np.linalg.norm(np.array(current_point).astype(
            float) - np.array(target_point).astype(float))

        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(float(ty) - float(cy), float(tx) - float(cx))
        in_sandtrap = is_in_sand_trap(
            current_point, self.sand_trap_matlab_polys, cache=self.map_points_in_sand_trap)

        target_sandtrap = is_in_sand_trap(
            target_point, self.sand_trap_matlab_polys, cache=self.map_points_in_sand_trap)

        splash_zone_poly_points = splash_zone(float(distance), float(
            angle), float(conf), self.skill, current_point, in_sandtrap, target_sandtrap)

        splash_in_poly = self.shapely_poly.contains(ShapelyPolygon(splash_zone_poly_points))

        splash_in_sand = False
        for sand_trap in self.sand_trap_shapely_polys:
            if sand_trap.intersects(ShapelyPolygon(splash_zone_poly_points)):
                splash_in_sand = True
                break

        return [splash_in_poly, splash_in_sand]

    def numpy_adjacent_and_dist(self, point: Tuple[float, float], conf: float, in_sandtrap: bool):
        cloc_distances = cdist(self.np_map_points, np.array(
            [np.array(point)]), 'euclidean')
        cloc_distances = cloc_distances.flatten()

        if in_sandtrap:
            distance_mask = cloc_distances <= self._max_sandtrap_ddist_ppf(conf)
            reachable_points = self.np_map_points[distance_mask]
            goal_distances = self.np_goal_dist[distance_mask]

        else:
            distance_mask = cloc_distances <= self._max_ddist_ppf(conf)
            reachable_points = self.np_map_points[distance_mask]
            goal_distances = self.np_goal_dist[distance_mask]

        return reachable_points, goal_distances

    def next_target(self, curr_loc: Tuple[float, float], goal: Point2D, conf: float) -> Union[None, Tuple[float, float]]:
        point_goal = float(goal.x), float(goal.y)
        heap = [ScoredPoint(curr_loc, point_goal, 0.0)]
        start_point = heap[0].point
        # Used to cache the best cost and avoid adding useless points to the heap
        best_cost = {tuple(curr_loc): 0.0}
        visited = set()
        points_checked = 0
        while len(heap) > 0:
            next_sp = heapq.heappop(heap)
            next_p = next_sp.point
            actual_cost = next_sp.actual_cost

            if next_p in visited:
                continue
            if next_sp.actual_cost > 10:
                continue

            visited.add(next_p)

            if np.linalg.norm(np.array(self.goal) - np.array(next_p)) <= 5.4 / 100.0:
                # All we care about is the next point
                # TODO: We need to check if the path length is <= 10, because if it isn't we probably need to
                #  reduce the conf and try again for a shorter path.
                while next_sp.previous.point != start_point:
                    next_sp = next_sp.previous
                return next_sp.point

            # Add adjacent points to heap
            shot_distance = 0 if next_sp.previous is None else np.linalg.norm(np.array(next_p) - np.array(next_sp.previous.point))
            start_point_in_sand = False if next_sp.previous is None else is_in_sand_trap(next_sp.previous.point, self.sand_trap_matlab_polys, cache=self.map_points_in_sand_trap)
            target_point_in_sand = is_in_sand_trap(next_p, self.sand_trap_matlab_polys, cache=self.map_points_in_sand_trap)

            if next_sp.previous is None:
                # target point is the starting point, no rolling
                next_p_after_rolling = next_p
            elif target_point_in_sand:
                # target point is in a sand trap, no rolling
                next_p_after_rolling = next_p
            elif not start_point_in_sand and shot_distance < 20:
                # putter shot, no rolling
                next_p_after_rolling = next_p
            else:
                next_p_after_rolling = roll(next_sp.previous.point, next_p, constants.extra_roll)

            reachable_points, goal_dists = self.numpy_adjacent_and_dist(
                next_p_after_rolling, conf, is_in_sand_trap(next_p_after_rolling, self.sand_trap_matlab_polys, cache=self.map_points_in_sand_trap))


            for i in range(len(reachable_points)):
                # shooting from @next_p -> @candidate_point
                candidate_point = tuple(reachable_points[i])
                shot_distance = np.linalg.norm(np.array(candidate_point) - np.array(next_p))
                start_point_in_sand = is_in_sand_trap(next_p, self.sand_trap_matlab_polys, cache=self.map_points_in_sand_trap)
                putter_shot = shot_distance < 20 and not start_point_in_sand
                new_actual_cost = actual_cost

                if candidate_point not in best_cost or best_cost[candidate_point] > new_point.actual_cost:
                    points_checked += 1

                    sand_penalty = 0
                    splash_in_poly, splash_in_sand = self.splash_zone_within_polygon(next_p, candidate_point, conf)

                    if not splash_in_poly:
                        continue
                    if splash_in_sand and not is_in_sand_trap(candidate_point, self.sand_trap_matlab_polys, cache=self.map_points_in_sand_trap):
                        sand_penalty = self.AVOID_SAND_PENALTY
                    if splash_in_sand and putter_shot:
                        new_actual_cost += self.PUTTER_OVER_SAND_PENALTY

                    goal_dist = goal_dists[i]
                    new_point = ScoredPoint(candidate_point, point_goal, new_actual_cost + 1, next_sp,
                                            goal_dist=goal_dist, skill=self.skill, sand_penalty=sand_penalty)

                    best_cost[new_point.point] = new_point.actual_cost
                    heapq.heappush(heap, new_point)

        # No path available
        return None

    def _initialize_map_points(self, goal: Tuple[float, float], golf_map: Polygon, sand_traps: List[sympy.Polygon]):
        np_map_points = [goal]  # storing the points as numpy array
        self.mpl_poly = sympy_poly_to_mpl(golf_map)
        self.shapely_poly = sympy_poly_to_shapely(golf_map)
        self.sand_trap_matlab_polys = [sympy_poly_to_mpl(
            sand_trap) for sand_trap in sand_traps]
        self.sand_trap_shapely_polys = [sympy_poly_to_shapely(
            sand_trap) for sand_trap in sand_traps]

        pp = list(poly_to_points(golf_map))
        for point in pp:
            # Use matplotlib here because it's faster than shapely for this calculation...
            if self.mpl_poly.contains_point(point):
                x, y = point
                np_map_points.append(np.array([x, y]))

        self.map_points_in_sand_trap = find_map_points_in_sand_trap(
            pp, self.sand_trap_matlab_polys)
        self.np_map_points = np.array(np_map_points)
        self.np_goal_dist = cdist(self.np_map_points, np.array(
            [np.array(self.goal)]), 'euclidean')
        self.np_goal_dist = self.np_goal_dist.flatten()

    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, sand_traps: List[sympy.Polygon], curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D, prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
        """Function which based n current game state returns the distance and angle, the shot must be played 

        Args:
            score (int): Your total score including current turn
            golf_map (sympy.Polygon): Golf Map polygon
            target (sympy.geometry.Point2D): Target location
            curr_loc (sympy.geometry.Point2D): Your current location
            prev_loc (sympy.geometry.Point2D): Your previous location. If you haven't played previously then None
            prev_landing_point (sympy.geometry.Point2D): Your previous shot landing location. If you haven't played previously then None
            prev_admissible (bool): Boolean stating if your previous shot was within the polygon limits. If you haven't played previously then None

        Returns:
            Tuple[float, float]: Return a tuple of distance and angle in radians to play the shot
        """

        if self.np_map_points is None:
            gx, gy = float(target.x), float(target.y)
            self.goal = float(target.x), float(target.y)
            self._initialize_map_points((gx, gy), golf_map, sand_traps)

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
            original_dist = np.linalg.norm(
                np.array(target_point) - current_point)
            v = np.array(target_point) - current_point
            # Unit vector pointing from current to target
            u = v / original_dist
            if original_dist >= 20.0 or is_in_sand_trap(tuple(current_point), self.sand_trap_matlab_polys, cache=self.map_points_in_sand_trap):
                roll_distance = original_dist / 20
                max_offset = roll_distance
                offset = 0
                prev_target = target_point
                while offset < max_offset and self.splash_zone_within_polygon(tuple(current_point), tuple(target_point), confidence)[0]:
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


# === Unit Tests ===

def test_reachable():
    current_point = Point2D(0, 0, evaluate=False)
    target_point = Point2D(0, 250, evaluate=False)
    player = Player(50, 0xdeadbeef, None)

    assert not player.reachable_point(current_point, target_point, 0.80)


def test_splash_zone_within_polygon():
    poly = Polygon((0, 0), (0, 300), (300, 300), (300, 0), evaluate=False)

    current_point = Point2D(0, 0, evaluate=False)

    # Just checking polygons inside and outside
    inside_target_point = Point2D(150, 150, evaluate=False)
    outside_target_point = Point2D(299, 100, evaluate=False)

    player = Player(50, 0xdeadbeef, None)
    assert player.splash_zone_within_polygon(
        player.splash_zone(current_point, inside_target_point, poly, 0.8))
    assert not player.splash_zone_within_polygon(
        player.splash_zone(current_point, outside_target_point, poly, 0.8))


def test_poly_to_points():
    poly = Polygon((0, 0), (0, 10), (10, 10), (10, 0))
    points = set(poly_to_points(poly))
    for x in range(1, 10):
        for y in range(1, 10):
            assert (x, y) in points
    assert len(points) == 81


def test_is_in_sand_trap():
    cases = [
        {
            "name": "point in one of the sand traps",
            "point": (5, 5),
            "sand_traps": [
                Polygon((4, 4), (4, 9), (9, 9), (9, 4)),
                Polygon((0, 0), (0, 4), (3, 0))
            ],
            "expect": True
        },
        {
            "name": "point not in any sand trap",
            "point": (10, 10),
            "sand_traps": [
                Polygon((4, 4), (4, 9), (9, 9), (9, 4)),
                Polygon((0, 0), (0, 4), (3, 0))
            ],
            "expect": False
        },
        {
            "name": "point in cache",
            "point": (5, 5),
            "sand_traps": [],  # empty to test if cache is checked
            "cache": {(5, 5)},
            "expect": True
        }
    ]

    for tc in cases:
        sand_trap_matplot_polys = [sympy_poly_to_mpl(
            poly) for poly in tc["sand_traps"]]
        cache = set() if "cache" not in tc else tc["cache"]
        ans = is_in_sand_trap(tc["point"], sand_trap_matplot_polys, cache)

        assert ans == tc["expect"]


def test_find_map_points_in_sand_trap():
    cases = [
        {
            "name": "no point in sand trap",
            "map_points": [(9.5, 9.5), (5., 5.), (5., 10.), (10., 5.), (10., 10.)],
            "sand_traps": [
                Polygon((0, 0), (0, 2), (2, 2), (2, 0))
            ],
            "expect": set()
        },
        {
            "name": "one point in sand trap",
            "map_points": [(9.5, 9.5), (5., 5.), (5., 10.), (10., 5.), (10., 10.)],
            "sand_traps": [
                Polygon((4, 4), (4, 9), (9, 9), (9, 4))
            ],
            "expect": {(5., 5.)}
        },
        {
            "name": "multiple points in sand trap",
            "map_points": [(9.5, 9.5), (5., 5.), (5., 10.), (10., 5.), (10., 10.)],
            "sand_traps": [
                Polygon((4, 4), (4, 9), (9, 9), (9, 4)),
                Polygon((9, 9), (9, 10), (10, 10), (10, 9))
            ],
            "expect": {(9.5, 9.5), (5., 5.)}
        }
    ]

    for tc in cases:
        sand_trap_paths = [sympy_poly_to_mpl(
            poly) for poly in tc["sand_traps"]]
        points_in_sand_trap = find_map_points_in_sand_trap(
            tc["map_points"], sand_trap_paths)
        assert points_in_sand_trap == tc["expect"]

def test_roll():
    cases = [
        {
            "current_point": (0.0, 0.0),
            "target_point": (10.0, 0.0),
            "rolling_factor": 0.1,
            "expect": (11.0, 0.0)
        },
        {
            "current_point": (0.0, 0.0),
            "target_point": (3.0, 3.0),
            "rolling_factor": 0.5,
            "expect": (4.5, 4.5)
        }
    ]

    for tc in cases:
        ans = roll(tc["current_point"], tc["target_point"], tc["rolling_factor"])
        assert ans == tc["expect"]
        assert type(ans[0]) == float and type(ans[1]) == float