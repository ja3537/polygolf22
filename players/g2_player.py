from dataclasses import dataclass
import os
import pickle
import numpy as np
import functools
import sympy
import logging
import heapq
from time import perf_counter
from scipy import stats as scipy_stats
import random
import matplotlib.pyplot as plt
from typing import Tuple, Iterator, List, Union
from sympy.geometry import Polygon, Point2D
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, shape as ShapelyShape, LineString as ShapelyLineString
from scipy.spatial.distance import cdist

# Cached distribution
DIST = scipy_stats.norm(0, 1)
SAND_DIST = scipy_stats.norm(0, 2)
X_STEP = 10
Y_STEP = 10


@functools.lru_cache()
def standard_ppf(conf: float) -> float:
    return DIST.ppf(conf)

@functools.lru_cache()
def sand_standard_ppf(conf: float) -> float:
    return SAND_DIST.ppf(conf)

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

def splash_zone(distance: float, angle: float, conf: float, skill: int, current_point: Tuple[float, float], current_point_in_sand: bool) -> np.array:
    conf_points = np.linspace(1 - conf, conf, 5)

    if (current_point_in_sand):
        distances = np.vectorize(sand_standard_ppf)(conf_points) * (distance / skill) + (distance/2)
        angles = np.vectorize(sand_standard_ppf)(conf_points) * (1/(2*skill)) + angle
    else:
        distances = np.vectorize(standard_ppf)(conf_points) * (distance / skill) + distance
        angles = np.vectorize(standard_ppf)(conf_points) * (1/(2*skill)) + angle
    scale = 1.1
    if distance <= 20 and not current_point_in_sand:
        scale = 1.0
    max_distance = distances[-1]*scale
    top_arc = spread_points(current_point, angles, max_distance, False)

    if distance > 20 or current_point_in_sand:
        min_distance = distances[0]
        bottom_arc = spread_points(current_point, angles, min_distance, True)
        return np.concatenate((top_arc, bottom_arc, np.array([top_arc[0]])))

    current_point = np.array([current_point])
    return np.concatenate((current_point, top_arc, current_point))

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
    v = sympy_poly.vertices
    v.append(v[0])
    return Path(v, closed=True)


def sympy_poly_to_shapely(sympy_poly: Polygon) -> ShapelyPolygon:
    """Helper function to convert sympy Polygon to shapely Polygon object"""
    v = list(sympy_poly.vertices)
    v.append(v[0])
    return ShapelyPolygon(v)

class ScoredPoint:
    """Scored point class for use in A* search algorithm"""
    def __init__(self, point: Tuple[float, float], goal: Tuple[float, float], actual_cost=float('inf'), previous=None, goal_dist=None, skill=50, in_sand=False):
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

        if in_sand:
            max_sand = max_dist/2
            if goal_dist <= max_sand:
                self._h_cost =  max_sand/max_sand + (goal_dist)/max_sand
            else:
                self._h_cost = max_sand/max_sand + (goal_dist-max_sand)/max_dist

        # given a point 21 and 20 away, we want to do the one with putter
        if goal_dist > 20:
            self._h_cost += 1

    
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
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger, golf_map: sympy.Polygon, start: sympy.geometry.Point2D, target: sympy.geometry.Point2D, sand_traps: list[sympy.Polygon], map_path: str, precomp_dir: str) -> None:
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
        self.mpl_poly = None
        self.shapely_poly = None
        self.goal = None
        self.prev_rv = None
        self.shapely_goal = None
        # Cached data
        max_dist = 200 + self.skill
        self.max_ddist = scipy_stats.norm(max_dist, max_dist / self.skill)
        self.max_sand_ddist = scipy_stats.norm(max_dist/2, 2 * max_dist / self.skill)

        self.map_points_is_sand = {}
        self.sand_traps = [sympy_poly_to_shapely(sympy_poly) for sympy_poly in sand_traps]

        self.current_shot_in_sand = None

        if self.np_map_points is None:
                gx, gy = float(target.x), float(target.y)
                self.goal = float(target.x), float(target.y)
                self.shapely_goal = ShapelyPoint(self.goal[0], self.goal[1])
                self._initialize_map_points((gx, gy), golf_map)
                # print(f"done init map with {len(self.np_map_points)} points")

        # Conf level
        self.conf = 0.99
        
    @functools.lru_cache()
    def _max_ddist_ppf(self, conf: float):
        return self.max_ddist.ppf(1.0 - conf)

    @functools.lru_cache()
    def _max__sand_ddist_ppf(self, conf: float):
        return self.max_sand_ddist.ppf(1.0 - conf)

    def reachable_point(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float) -> bool:
        """Determine whether the point is reachable with confidence [conf] based on our player's skill"""
        if type(current_point) == Point2D:
            current_point = tuple(current_point)
        if type(target_point) == Point2D:
            target_point = tuple(target_point)

        current_point = np.array(current_point).astype(float)
        target_point = np.array(target_point).astype(float)

        return np.linalg.norm(current_point - target_point) <= self._max_ddist_ppf(conf) if not self.is_in_sand(current_point) else self._max__sand_ddist_ppf(conf)
    
    def splash_zone_within_polygon(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float) -> bool:
        if type(current_point) == Point2D:
            current_point = tuple(Point2D)

        if type(target_point) == Point2D:
            target_point = tuple(Point2D)

        this_in_sand = self.is_in_sand(current_point)

        distance = np.linalg.norm(np.array(current_point).astype(float) - np.array(target_point).astype(float)) * (2 if this_in_sand else 1)
        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(float(ty) - float(cy), float(tx) - float(cx))
        splash_zone_poly_points = splash_zone(float(distance), float(angle), float(conf), self.skill, current_point, this_in_sand)
        shapely_splash_zone_poly_points = ShapelyPolygon(splash_zone_poly_points)

        if self.shapely_poly.contains(shapely_splash_zone_poly_points):
            if not self.is_in_sand(target_point) and not shapely_splash_zone_poly_points.contains(self.shapely_goal):
                total_overlap = sum([shapely_splash_zone_poly_points.intersection(sand_trap).area for sand_trap in self.sand_traps if shapely_splash_zone_poly_points.intersects(self.shapely_poly)])
                return total_overlap/shapely_splash_zone_poly_points.area <= 1 - conf
            return True
        return False

    def numpy_adjacent_and_dist(self, point: Tuple[float, float], conf: float):
        cloc_distances = cdist(self.np_map_points, np.array([np.array(point)]), 'euclidean')
        cloc_distances = cloc_distances.flatten()
        distance_mask = cloc_distances <= (self._max_ddist_ppf(conf) if not self.is_in_sand(point) else self._max__sand_ddist_ppf(conf))

        reachable_points = self.np_map_points[distance_mask]
        goal_distances = self.np_goal_dist[distance_mask]

        return reachable_points, goal_distances

    def next_target(self, curr_loc: Tuple[float, float], goal: Point2D, conf: float) -> Union[None, Tuple[float, float]]:
        # last_checked_time = perf_counter()
        point_goal = float(goal.x), float(goal.y)
        heap = [ScoredPoint(curr_loc, point_goal, 0.0, in_sand=self.is_in_sand(curr_loc))]
        start_point = heap[0].point
        # Used to cache the best cost and avoid adding useless points to the heap
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
            if next_sp.actual_cost > 0 and not self.splash_zone_within_polygon(next_sp.previous.point, next_p, conf): #check if shooting from prev to here will land in bounds
                if next_p in best_cost:
                    del best_cost[next_p]
                continue
            visited.add(next_p)

            if np.linalg.norm(np.array(self.goal) - np.array(next_p)) <= 5.4 / 100.0:
                # All we care about is the next point
                # TODO: We need to check if the path length is <= 10, because if it isn't we probably need to
                #  reduce the conf and try again for a shorter path.

                path_length = 0
                while next_sp.previous.point != start_point:
                    next_sp = next_sp.previous
                    path_length += 1

                return next_sp.point, path_length
            
            # Add adjacent points to heap
            reachable_points, goal_dists = self.numpy_adjacent_and_dist(next_p, conf)
            for i in range(len(reachable_points)):
                candidate_point = tuple(reachable_points[i])
                goal_dist = goal_dists[i]
                new_point = ScoredPoint(candidate_point, point_goal, next_sp.actual_cost + 1, next_sp,
                                        goal_dist=goal_dist, skill=self.skill, in_sand=self.is_in_sand(candidate_point))

                if candidate_point not in best_cost or best_cost[candidate_point] > new_point.actual_cost:
                    # points_checked += 1
                    # if not self.splash_zone_within_polygon(new_point.previous.point, new_point.point, conf):
                    #     continue
                    best_cost[new_point.point] = new_point.actual_cost
                    heapq.heappush(heap, new_point)

        # No path available
        return None, None

    def _initialize_map_points(self, goal: Tuple[float, float], golf_map: Polygon):
        # Storing the points as numpy array
        np_map_points = [goal]
        map_points = [goal]
        self.mpl_poly = sympy_poly_to_mpl(golf_map)
        self.shapely_poly = sympy_poly_to_shapely(golf_map)
        pp = list(poly_to_points(golf_map))

        np_map_points = [np.array([x, y]) for x, y in pp if self.mpl_poly.contains_point((x, y))]

        xmin, ymin, xmax, ymax = golf_map.bounds
        add_start = perf_counter()
        while (len(np_map_points) < 4000) and perf_counter() - add_start < 30:
            x, y = np.random.uniform(xmin, xmax, 5000), np.random.uniform(ymin, ymax, 5000)
            np_map_points += [np.array(point) for point in np.array([x, y], dtype=int).T if self.mpl_poly.contains_point(point)]
        
        np_map_points.insert(0, goal)
        # self.map_points = np.array(map_points)
        self.np_map_points = np.array(np_map_points)
        self.np_goal_dist = cdist(self.np_map_points, np.array([np.array(self.goal)]), 'euclidean')
        self.np_goal_dist = self.np_goal_dist.flatten()

    def is_in_sand(self, point: sympy.geometry.Point2D):
        if (type(point) == np.ndarray):
            point = Point2D(point[0], point[1])
        if (point not in self.map_points_is_sand):
            shapelyPoint = ShapelyPoint(point[0], point[1])
            self.map_points_is_sand[point] = any(s.contains(shapelyPoint) for s in self.sand_traps)
        return self.map_points_is_sand[point]


    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, sand_traps: list[sympy.Polygon], curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D, prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
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
        self.turn_start = perf_counter()
        self.score = score
        if self.np_map_points is None:
            gx, gy = float(target.x), float(target.y)
            self.goal = float(target.x), float(target.y)
            self.shapely_goal = ShapelyPoint(self.goal[0], self.goal[1])
            self._initialize_map_points((gx, gy), golf_map) ## initrialize at runtime? multi thread it?

        # Optimization to retry missed shots
        if not prev_admissible and self.prev_rv is not None:
            return self.prev_rv

        self.current_shot_in_sand = self.is_in_sand(curr_loc)

        confidence_intervals = [.95, .99, .75, .6, .85]
        
        target_points = {}
        target_point = None
        confidence = self.conf + 0.1
        cl = float(curr_loc.x), float(curr_loc.y)
        target_path_length = None

        for confidence in confidence_intervals:
            # print(f"turn # {score} searching with {confidence} confidence")
            target_point, target_path_length = self.next_target(cl, target, confidence)
            if target_point is None:
                continue
            target_points[confidence] = {
                "target_point": target_point,
                "target_path_length": target_path_length,
                "confidence": confidence
            }

            if (tuple(target_point) == self.goal):
                break
            if ((perf_counter() - self.turn_start)*(target_path_length + score)) > 540:
                break
        
        if len(target_points) == 0:
            return None
        if (tuple(target_point) != self.goal):
            smallest_path = min([target["target_path_length"] for target in target_points.values()])
            max_confidence_of_smallest_path = max([target["confidence"] for target in target_points.values() if target["target_path_length"] == smallest_path])
            target_point = target_points[max_confidence_of_smallest_path]["target_point"]

        # fixup target
        current_point = np.array(tuple(curr_loc)).astype(float)

        print(target_points)
        if tuple(target_point) == self.goal or np.linalg.norm(np.array(target_point) - current_point) < 20:
            original_dist = np.linalg.norm(np.array(target_point) - current_point)
            v = np.array(target_point) - current_point
            # Unit vector pointing from current to target
            u = v / original_dist

            if original_dist >= 20.0 or self.current_shot_in_sand:
                roll_distance = original_dist /  20
                max_offset = roll_distance
                offset = 0
                prev_target = target_point
                while offset < max_offset and self.splash_zone_within_polygon(tuple(current_point), target_point, confidence):
                    offset += 1
                    dist = original_dist - offset
                    prev_target = target_point
                    target_point = current_point + u * dist
                target_point = prev_target
                # print(f"old target {np.linalg.norm(np.array(target_point) - current_point)*1.1*u + current_point}")


                # find dist such that dist * 1.1 >= goal
                # find dist such that it lands between dist and goal most of the time
                # does this mean just the half way point??
                # current_point + somedistance* 1.1 * u == self.goal
                point_to_aim = self.goal - current_point
                perfect_distance_for_rolling = np.array([point_to_aim[0]/(1.1*u[0]), point_to_aim[1]/(1.1*u[1])])
                # print(f"perfect distance for rolling {perfect_distance_for_rolling}")
                # find the distance between the current point and the perfect point
                # print(f"1.1 of perfect dist landing spot {current_point + perfect_distance_for_rolling*(1.1*u)}")
                perfect_landing_spot = current_point + perfect_distance_for_rolling*u

                distance_from_land_to_goal = np.linalg.norm(self.goal - np.array(perfect_landing_spot))
                # print(f"distance from land to goal {distance_from_land_to_goal}")
                point_to_land = perfect_landing_spot + (u * distance_from_land_to_goal/2)
                #check if point is good splash zone
                max_tries = 0
                while not self.splash_zone_within_polygon(tuple(current_point), point_to_land, confidence) and max_tries < 5:
                    distance_to_offset = np.linalg.norm(self.goal - np.array(point_to_land))/2
                    point_to_land += u * distance_to_offset
                    max_tries += 1
                
                # print(f"new target = {np.linalg.norm(np.array(point_to_land) - current_point)*1.1*u + current_point}")
                target_point = point_to_land
 
                
                line = ShapelyLineString([self.goal, target_point])
                count = 0
                while any(line.intersects(s) for s in self.sand_traps) and count < 3:
                    count += 1
                    distance_between_points = np.linalg.norm(self.goal - np.array(target_point))
                    target_point += distance_between_points/2 * u
                    line = ShapelyLineString([self.goal, target_point])
            else:

                line = ShapelyLineString([self.goal, current_point])
                past_point = current_point + u * 20
                line_past_to_goal =  ShapelyLineString([self.goal, past_point])
                # if putt would run into sand, just go to goal but in the air
                if any(line.intersects(s) for s in self.sand_traps) and self.splash_zone_within_polygon(tuple(current_point), past_point, 0.95) and not any(line_past_to_goal.intersects(s) for s in self.sand_traps):
                    distance = 20 
                else:
                    conf = 0.95
                    # TODO: angles = np.vectorize(standard_ppf)(conf_points) * (1/(2*skill)) + angle

                    #min distance =  np.vectorize(standard_ppf)(1-conf)* (distance / self.skill)+ distance ??
                    # np.vectorize(standard_ppf)(1-conf) * (distance / self.skill) + distance = original_dist
                    # mindistance >= original_dist 
                    # 95% of the time we hit at least original distance (aka goal)
                    distance = min(19.9999, original_dist / (np.vectorize(standard_ppf)(1-conf) * (1/self.skill) + 1))
                    # print(current_point)
                    # print(f"new point {current_point + u * distance}")
                    # print(f"old point {self.goal + u * 0.54}")
                target_point = current_point + u * distance





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
    poly = Polygon((0,0), (0, 300), (300, 300), (300, 0), evaluate=False)

    current_point = Point2D(0, 0, evaluate=False)

    # Just checking polygons inside and outside
    inside_target_point = Point2D(150, 150, evaluate=False)
    outside_target_point = Point2D(299, 100, evaluate=False)

    player = Player(50, 0xdeadbeef, None)
    assert player.splash_zone_within_polygon(current_point, inside_target_point, poly, 0.8)
    assert not player.splash_zone_within_polygon(current_point, outside_target_point, poly, 0.8)


def test_poly_to_points():
    poly = Polygon((0,0), (0, 10), (10, 10), (10, 0))
    points = set(poly_to_points(poly))
    for x in range(1, 10):
        for y in range(1, 10):
            assert (x,y) in points
    assert len(points) == 81
