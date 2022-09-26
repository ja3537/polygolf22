import os
import pickle
import numpy as np
import functools
import sympy
import logging
import heapq
from scipy import stats as scipy_stats


from typing import Tuple, Iterator, List, Union
from sympy.geometry import Polygon, Point2D
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, LineString as ShapelyLine
from scipy.spatial.distance import cdist

# Cached distribution
DIST = scipy_stats.norm(0, 1)
X_STEP = 5.0
Y_STEP = 5.0
NEARBY_DIST = 100


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

def splash_zone(distance: float, angle: float, conf: float, skill: int, current_point: Tuple[float, float], target_trapped=False) -> np.array:
    conf_points = np.linspace(1 - conf, conf, 5)
    distances = np.vectorize(standard_ppf)(conf_points) * (distance / skill) + distance
    angles = np.vectorize(standard_ppf)(conf_points) * (1/(2*skill)) + angle
    scale = 1.1
    if target_trapped or distance <= 20:
        scale = 1.0
    max_distance = distances[-1]*scale
    top_arc = spread_points(current_point, angles, max_distance, False)

    if distance > 20:
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
    v = list(sympy_poly.vertices)
    v.append(v[0])
    return Path(v, closed=True)


def sympy_poly_to_shapely(sympy_poly: Polygon) -> ShapelyPolygon:
    """Helper function to convert sympy Polygon to shapely Polygon object"""
    v = list(sympy_poly.vertices)
    v.append(v[0])
    return ShapelyPolygon(v)

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
        max_dist = standard_ppf(0.99) * (max_target_dist / skill) + max_target_dist
        max_dist *= 1.10
        self._h_cost = (sand_penalty + goal_dist) / max_dist

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
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger, golf_map: sympy.Polygon, start: sympy.geometry.Point2D, target: sympy.geometry.Point2D, sand_traps: List[sympy.geometry.Point2D], map_path: str, precomp_dir: str) -> None:
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
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.np_map_points = None
        self.shapely_poly = None
        self.goal = None
        self.visited = set()
        self.new_visited = set()
        self.mode = 'a_star'
        self.prev_rv = None

        # Cached data
        max_dist = 200 + self.skill
        self.max_ddist = scipy_stats.norm(max_dist, max_dist/self.skill)
        self.sand_max_ddist = scipy_stats.norm(max_dist/2, max_dist/self.skill*2)

        self.nearby_ddist = scipy_stats.norm(NEARBY_DIST, NEARBY_DIST/self.skill)
        self.sand_nearby_ddist = scipy_stats.norm(NEARBY_DIST/2, NEARBY_DIST/self.skill*2)

        # Conf level
        self.conf = 0.95
        if self.skill < 40:
            self.conf = 0.75

        self.num_trials = 1000
        self.prev_loc = None

    def _initialize_map_points(self, goal: Tuple[float, float], golf_map: Polygon, sand_traps):
        # Storing the points as numpy array
        np_map_points = [goal]
        map_points = [goal]
        self.mpl_poly = sympy_poly_to_mpl(golf_map)
        self.mpl_sand_polys = [sympy_poly_to_mpl(trap) for trap in sand_traps]
        self.shapely_sand_polys = [sympy_poly_to_shapely(trap) for trap in sand_traps]
        self.shapely_poly = sympy_poly_to_shapely(golf_map)
        pp = list(poly_to_points(golf_map))
        sand_penalty = [0]
        for point in pp:
            # Use matplotlib here because it's faster than shapely for this calculation...
            if self.mpl_poly.contains_point(point):
                x, y = point
                np_map_points.append(np.array([x, y]))
                trapped = False
                for trap_i, trap in enumerate(self.mpl_sand_polys):
                    if trap.contains_point(point):
                        penalty = self.shapely_sand_polys[trap_i].exterior.distance(ShapelyPoint(point))
                        sand_penalty.append(penalty)
                        trapped = True
                        break
                if not trapped: sand_penalty.append(0)
        
        self.np_sand_penalty = np.array(sand_penalty)
        self.np_map_points = np.array(np_map_points)
        self.np_goal_dist = cdist(self.np_map_points, np.array([np.array(self.goal)]), 'euclidean')
        self.np_goal_dist = self.np_goal_dist.flatten()

        #print(self.np_map_points.shape, self.np_sand_penalty.shape, self.np_goal_dist.shape)

    @functools.lru_cache()
    def _max_ddist_ppf(self, conf: float):
        return self.max_ddist.ppf(1.0 - conf)

    @functools.lru_cache()
    def _sand_max_ddist_ppf(self, conf: float):
        return self.sand_max_ddist.ppf(1.0 - conf)

    @functools.lru_cache()
    def _nearby_ddist_ppf(self, conf: float):
        return self.nearby_ddist.ppf(1.0 - conf)

    @functools.lru_cache()
    def _sand_nearby_ddist_ppf(self, conf: float):
        return self.sand_nearby_ddist.ppf(1.0 - conf)

    def splash_zone_within_polygon(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float, target_trapped=False) -> bool:
        if type(current_point) == Point2D:
            current_point = tuple(Point2D)

        if type(target_point) == Point2D:
            target_point = tuple(Point2D)

        distance = np.linalg.norm(np.array(current_point).astype(float) - np.array(target_point).astype(float))
        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(float(ty) - float(cy), float(tx) - float(cx))
        splash_zone_poly_points = splash_zone(float(distance), float(angle), float(conf), self.skill, current_point, target_trapped=target_trapped)
        return self.shapely_poly.contains(ShapelyPolygon(splash_zone_poly_points))

    def numpy_adjacent_and_dist(self, point: Tuple[float, float], conf: float, mode='max', trapped=False):
        cloc_distances = cdist(self.np_map_points, np.array([np.array(point)]), 'euclidean')
        cloc_distances = cloc_distances.flatten()
        if trapped:
            if mode == 'max':
                distance_mask = cloc_distances <= self._sand_max_ddist_ppf(conf)
            elif mode == 'nearby':
                distance_mask = cloc_distances <= self._sand_nearby_ddist_ppf(conf)
        else:
            if mode == 'max':
                distance_mask = cloc_distances <= self._max_ddist_ppf(conf)
            elif mode == 'nearby':
                distance_mask = cloc_distances <= self._nearby_ddist_ppf(conf)

        reachable_points = self.np_map_points[distance_mask]
        goal_distances = self.np_goal_dist[distance_mask]
        sand_penalties = self.np_sand_penalty[distance_mask]

        if not trapped:
            # check if there's sand blocking a putter shot
            putt_shot_indices = np.where(cloc_distances < 20)[0]
            for i in putt_shot_indices:
                line = ShapelyLine([point, tuple(self.np_map_points[i])])
                if any([line.intersects(trap) for trap in self.shapely_sand_polys]):
                    distance_mask[i] = False

        reachable_points = self.np_map_points[distance_mask]
        goal_distances = self.np_goal_dist[distance_mask]
        sand_penalties = self.np_sand_penalty[distance_mask]

        return reachable_points, goal_distances, sand_penalties

    def next_target_greedy(self, curr_loc: Tuple[float, float], goal: Point2D, conf: float) -> Union[None, Tuple[float, float]]:
        if self.prev_loc != curr_loc:
            self.visited = self.visited.union(self.new_visited)
            self.prev_loc = curr_loc
        self.new_visited = set()
        trapped = any([trap.contains_point(curr_loc) for trap in self.mpl_sand_polys])
        point_goal = float(goal.x), float(goal.y)
        reachable_points, goal_dists, _ = self.numpy_adjacent_and_dist(curr_loc, conf, trapped=trapped)
        sorted_idx = np.argsort(goal_dists)
        goal_dists = goal_dists[sorted_idx]
        reachable_points = reachable_points[sorted_idx]
        #selected_idx = np.random.choice(np.arange(min(reachable_points.shape[0], self.num_trials)))
        for i in range(min(reachable_points.shape[0], self.num_trials)):
            landing_point = tuple(reachable_points[i])
            if not landing_point in self.visited and self.splash_zone_within_polygon(curr_loc, landing_point, conf):
                self.new_visited.add(landing_point)
                if np.linalg.norm(np.array(landing_point) - np.array(point_goal)) > NEARBY_DIST:
                    nearby_points = self.numpy_adjacent_and_dist(curr_loc, conf, mode='nearby', trapped=trapped)[0]
                    for j in range(nearby_points.shape[0]):
                        self.new_visited.add(tuple(nearby_points[j]))
                return landing_point

        # No path available
        return None

    def next_target(self, curr_loc: Tuple[float, float], goal: Point2D, conf: float) -> Union[None, Tuple[float, float]]:
        trapped = any([trap.contains_point(curr_loc) for trap in self.mpl_sand_polys])
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

            if next_p in visited:
                continue
            if next_sp.actual_cost > 10:
                continue
            if next_sp.actual_cost > 0:
                target_trapped = any([trap.contains_point(next_p) for trap in self.mpl_sand_polys])
                if not self.splash_zone_within_polygon(next_sp.previous.point, next_p, conf, target_trapped=target_trapped):
                    if next_p in best_cost:
                        del best_cost[next_p]
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
            reachable_points, goal_dists, sand_penalties = self.numpy_adjacent_and_dist(next_p, conf, trapped=trapped)
            for i in range(len(reachable_points)):
                candidate_point = tuple(reachable_points[i])
                goal_dist = goal_dists[i]
                sand_penalty = sand_penalties[i]
                new_point = ScoredPoint(candidate_point, point_goal, next_sp.actual_cost + 1, next_sp,
                                        goal_dist=goal_dist, skill=self.skill, sand_penalty=sand_penalty)
                if candidate_point not in best_cost or best_cost[candidate_point] > new_point.actual_cost:
                    points_checked += 1
                    # if not self.splash_zone_within_polygon(new_point.previous.point, new_point.point, conf):
                    #     continue
                    best_cost[new_point.point] = new_point.actual_cost
                    heapq.heappush(heap, new_point)

        # No path available
        return None

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

            #print(f"searching with {confidence} confidence")
            target_point = self.next_target_greedy(cl, target, confidence) if self.mode == 'greedy' else self.next_target(cl, target, confidence)
            confidence -= 0.05

        # fixup target
        current_point = np.array(tuple(curr_loc)).astype(float)
        trapped = any([trap.contains_point(current_point) for trap in self.mpl_sand_polys])
        if tuple(target_point) == self.goal:
            original_dist = np.linalg.norm(np.array(target_point) - current_point)
            v = np.array(target_point) - current_point
            # Unit vector pointing from current to target
            u = v / original_dist
            if original_dist >= 20.0:
                max_offset = original_dist / 20
                offset = 0
                prev_target = target_point
                while offset < max_offset and self.splash_zone_within_polygon(tuple(current_point), target_point, confidence):
                    offset += 1
                    dist = original_dist - offset
                    prev_target = target_point
                    target_point = current_point + u * dist
                target_point = prev_target + u * 2 # shoot further in hope that the ball would roll into the goal
            elif (not trapped) and self.skill >= 70:
                target_point += u


        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(ty - cy, tx - cx)

        rv = curr_loc.distance(Point2D(target_point, evaluate=False)), angle
        self.prev_rv = rv
        return rv



