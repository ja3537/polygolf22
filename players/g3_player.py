import numpy as np
import pandas as pd
import sympy
import random
import shapely.geometry, shapely.ops
import sklearn.cluster
import logging
from typing import Tuple, List
import matplotlib.pyplot as plt

POINTS_PER_REGION = 25

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

        self.skill = skill
        self.rng = rng
        self.logger = logger

        self.shapely_map = shapely.geometry.Polygon(golf_map.vertices)
        self.shapely_sand_traps = [shapely.geometry.Polygon(st.vertices) for st in sand_traps]
        self.split_polygon(self.shapely_map, self.shapely_sand_traps, 50)


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

        required_dist = curr_loc.distance(target)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0
        distance = sympy.Min(200+self.skill, required_dist/roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        return (distance, angle)

    def split_polygon(self, golf_map: sympy.Polygon, sand_traps: List[shapely.geometry.Polygon], regions: int) -> List[shapely.geometry.Polygon]:
        """ Split a given Golf Map into regions of roughly equal size.
        Based on an algorithm described by Paul Ramsey: http://blog.cleverelephant.ca/2018/06/polygon-splitting.html

        Args:
            golf_map (shapely.geometry.Polygon): The Golf Map to split into equal sized regions
            sand_traps (shapely.geometry.Polygon): A list of Sand Traps contained within the Golf Map 
            regions (int): The number of roughly equal sized regions to split the map into

        Returns:
            List[shapely.geometry.Polygon]: Returns a list of polygons, each representing a roughly equal sized region of the given map

        """

        # Naively insert holes into the given map where there are sand traps
        golf_map_with_holes = shapely.geometry.Polygon(golf_map.exterior.coords, [list(st.exterior.coords) for st in sand_traps])

        # Generate random points within the bounds of the given map
        # (based on https://gis.stackexchange.com/questions/207731/generating-random-coordinates-in-multipolygon-in-python)
        points = []
        min_x, min_y, max_x, max_y = golf_map_with_holes.bounds
        while len(points) < POINTS_PER_REGION*regions:
            pt = shapely.geometry.Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if golf_map_with_holes.contains(pt):
                points.append(pt)

        # Cluster the random points into groups using kmeans
        points_df = pd.DataFrame([[pt.x, pt.y] for pt in points], columns=['x', 'y'])
        kmeans = sklearn.cluster.KMeans(n_clusters=regions, init='k-means++').fit(points_df)

        # Generate a voronoi diagram from the centers of the generated regions
        center_points = shapely.geometry.MultiPoint(kmeans.cluster_centers_)
        regions = shapely.ops.voronoi_diagram(center_points)

        # Intersect the generated regions with the given map
        regions = [region.intersection(golf_map_with_holes) for region in regions.geoms]

        # Plot the random points, cluster centers, and voronoi regions
        plt.plot(*golf_map_with_holes.exterior.xy)
        plt.scatter([pt.x for pt in points], [pt.y for pt in points], s=5)
        plt.scatter([pt[0] for pt in kmeans.cluster_centers_], [pt[1] for pt in kmeans.cluster_centers_], color='red')
        for region in regions:
            plt.plot(*region.exterior.xy)
        plt.show()

        return regions

