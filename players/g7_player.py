import numpy as np
import scipy
import sympy
import functools
import logging
from typing import Tuple, Iterator
from scipy import stats as scipy_stats
from sympy.geometry import Polygon, Point2D
from shapely.geometry import Polygon as ShapelyPolygon

STEP = 1.0 # chunk size == 1m

DIST = scipy_stats.norm(0, 1)
X_STEP = 1.0
Y_STEP = 1.0

@functools.lru_cache()
def standard_ppf(conf: float) -> float:
    return DIST.ppf(conf)

def polygon_to_points(golf_map: sympy.Polygon) -> Iterator[Tuple[float, float]]:
    """
    This function takes in the polygon golf map and returns an iterator for the
    points on a lattice with distance STEP. We ignore the edges of the map
    where there is only water.
    """
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')

    for vertex in golf_map.vertices:
        x, y = float(vertex.x), float(vertex.y)

        x_min = min(x_min, x)
        x_max = min(x_max, x)
        y_min = min(y_min, y)
        y_max = min(y_max, y)
    
    step = STEP
    x_curr, y_curr = x_min, y_min
    while x_curr < x_max:
        while y_curr < y_max:
            yield float(x_curr), float(y_curr)
            y_curr += step
        y_curr = y_min
        x_curr += step


def sympy_polygon_to_shapely(polygon: sympy.Polygon) -> ShapelyPolygon:
    """
    Helper function that converts a sympy polygon to a shapely polygon
    """
    vertices = polygon.vertices
    vertices.append(vertices[0])
    return ShapelyPolygon(vertices)

def spread_points(current_point, angles: np.array, distance, reverse) -> np.array:
    curr_x, curr_y = current_point
    if reverse:
        angles = np.flip(angles)
    xs = np.cos(angles) * distance + curr_x
    ys = np.sin(angles) * distance + curr_y
    return np.column_stack((xs, ys))

def splash_zone(distance: float, angle: float, conf: float, skill: int, current_point: Tuple[float, float]) -> np.array:
    conf_points = np.linspace(1 - conf, conf, 5)
    distances = np.vectorize(standard_ppf)(conf_points) * (distance / skill) + distance
    angles = np.vectorize(standard_ppf)(conf_points) * (1/(2*skill)) + angle
    scale = 1.1
    if distance <= 20:
        scale = 1.0
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
    def __init__(self, point: Tuple[float, float], goal: Tuple[float, float], actual_cost=float('inf'), previous=None, goal_dist=None, skill=50):
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
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger, golf_map: sympy.Polygon, start: sympy.geometry.Point2D, target: sympy.geometry.Point2D, sand_traps: list[sympy.geometry.Point2D], map_path: str, precomp_dir: str) -> None:
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

        self.conf = 0.95 # confidence level, we might want to decrease this for low skill players
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.target = None
        self.np_points = None
        self.shapely_polygon = None
        self.shapely_polygon_trap = None
        
        max_distance = 200 + self.skill
        self.max_ddist = scipy_stats.norm(max_distance, max_distance / self.skill)

    def polygon_to_np_points(self, target: Tuple[float, float], golf_map: sympy.Polygon, sand_traps: list[sympy.geometry.Point2D]):
        """
        Gets the points within the polygon and stores them in a numpy array 
        along with their distances to the target.
        
        Added extra number at the end of tuple to denote wether the point is within a sandtrap
        or just within the grass.
        """
        points = [target]
        trap_points = []
        self.shapely_polygon = sympy_polygon_to_shapely(golf_map)
        for trap in sand_traps:
            self.shapely_polygon_trap = sympy_polygon_to_shapely(trap)
            polygon_points = polygon_to_points(trap)
            for point in polygon_points:
                if self.shapely_polygon_trap.contains(point):
                    x, y = point
                    trap_points.append(np.array([x,y,1]))
            #trap_points.append(polygon_to_points(trap))

        polygon_points = polygon_to_points(golf_map)
        for point in polygon_points:
            if self.shapely_polygon.contains(point) not in trap_points:
                x, y = point
                points.append(np.array([x, y,0]))
        
        self.np_points = np.array(points)
        self.np_dist_to_target = scipy.spatial.distance.cdist(self.np_points, np.array([np.array(self.target)]), 'euclidean')
        self.np_dist_to_target = self.np_dist_to_target.flatten()

    def reachable(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float) -> bool:
        if type(current_point) == Point2D:
            current_point = tuple(current_point)
        if type(target_point) == Point2D:
            target_point = tuple(target_point)
            
        current_point = np.array(current_point).astype(float)
        target_point = np.array(target_point).astype(float)
        
        distance = current_point - target_point
        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(float(ty) - float(cy), float(tx) - float(cx))
        
        return np.linalg.norm(distance) <= self._max_ddist_ppf(conf)
        
    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, sand_traps: list[sympy.geometry.Point2D], curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D, prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
        """Function which based on current game state returns the distance and angle, the shot must be played

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
        
        
        required_dist = curr_loc.distance(target)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0
        distance = sympy.Min(200+self.skill, required_dist/roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        return (distance, angle)