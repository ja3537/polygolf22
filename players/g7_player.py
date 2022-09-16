import numpy as np
import scipy
import sympy
import logging
from typing import Tuple, Iterator
from shapely.geometry import Polygon as ShapelyPolygon

STEP = 1.0 # chunk size == 1m

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


    def polygon_to_np_points(self, target: Tuple[float, float], golf_map: sympy.Polygon):
        """
        Gets the points within the polygon and stores them in a numpy array 
        along with their distances to the target.
        """
        points = [target]
        self.shapely_polygon = sympy_polygon_to_shapely(golf_map)

        polygon_points = polygon_to_points(golf_map)
        for point in polygon_points:
            if self.shapely_polygon.contains(point):
                x, y = point
                points.append(np.array([x, y]))
        
        self.np_points = np.array(points)
        self.np_dist_to_target = scipy.spatial.distance.cdist(self.np_points, np.array([np.array(self.target)]), 'euclidean')
        self.np_dist_to_target = self.np_dist_to_target.flatten()


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