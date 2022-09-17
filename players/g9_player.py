import numpy as np
import sympy
import logging
from typing import Tuple, List
from shapely.geometry import Polygon, Point, LineString


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


        self.shapely_poly = Polygon([(p.x, p.y) for p in golf_map.vertices])
        self.shapely_edges = LineString(list(self.shapely_poly.exterior.coords))
        self.start_pt = Point(start[0], start[1])
        self.end_pt = Point(target[0], target[1])


        self.sand_traps = []
        for trap in sand_traps:
            self.sand_traps.append(Polygon([(p.x, p.y) for p in trap.vertices]))


    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, sand_traps: List[sympy.geometry.Point2D], curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D, prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
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


        print("Testing")
        print(self.start_pt)

        for trap in self.sand_traps:
            print("Trap : ")
            print(trap)


    def in_sand_trap(self, x, y):
        point = Point(x, y)
        for trap in self.sand_traps:
            if trap.contains(point):
                return True
        return False