import numpy as np
import sympy
import logging
from typing import Tuple, Iterator

STEP = 1.0 # chunk size == 1m

def make_grid(polygon_map: sympy.Polygon) -> Iterator[Tuple[float, float]]:
    """
    This function takes in the polygon golf map and returns an iterator for the
    points on a lattice with distance STEP. We ignore the edges of the map
    where there is only water.
    """
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')

    for vertex in polygon_map.vertices:
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

        self.conf = 0.95 # confidence level
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.np_grid = None


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