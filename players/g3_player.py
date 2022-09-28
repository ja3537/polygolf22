import numpy as np
import pandas as pd
import sympy
import random
import shapely.geometry, shapely.ops
import logging
import matplotlib.pyplot as plt
from math import floor
import os
import pickle
import functools
import heapq
from scipy import stats as scipy_stats
from os import listdir
from os.path import isfile, join
from typing import Tuple, Iterator, List, Union, Dict
from sympy.geometry import Polygon, Point2D
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from scipy.spatial.distance import cdist
from faiss import Kmeans
from polylabel import polylabel


POINT_SPACING = 1

""" Constants to adjust confidence based on skill level """
HIGH_SKILL_CONFIDENCE = 0.90
LOW_SKILL_CONFIDENCE = 0.85
SKILL_CONFIDENCE_THRESHHOLD = 50
PUTTING_CONFIDENCE = 0.1

""" Constants to adjust heuristic cost for scored points in A* search """
FIXED_SANDTRAP_COST = 0.5                   # estimated additional shots required by entering sand trap
REMAINING_SHOTS_WEIGHT = 2                  # weight of estimated number of remaining shots from point (higher avoid points far from goal)
SAND_TRAP_WEIGHT = 0.5                        # weight of point being in a sand trap (high avoids entering sand traps)
REACHABLE_POINTS_WEIGHT = 0.01                 # weight of nearby sand vs. grass (high avoids areas with lots of nearby grass)
SAND_TRAP_IN_SPLASH_ZONE_WEIGHT = 3         # weight of target splash zone containing sand traps (high avoids splash zones containing sand traps)

""" Constants to adjust backup logic that handles rolling """
BACKUP_INCREMENT = 0.1
UNDO_BACKUP_INCREMENTS = 50                 # if we backup into a sand trap, how much backing up should we undo to avoid the sand
EXTRA_BACKUP_PCT_TO_AVOID_SAND_TRAPS = 0.1  # how much further back should we go to avoid a sand trap (going further back means not reaching our target)
METERS_TO_OVERSHOOT_HOLE = 1                # when aiiming for the hole, how many meters past should we try to overshoot so that we roll in
PCT_SPLASH_ZONE_IN_MAP_TO_BACK_UP = 0.96    # how much of the target splash zone needs to be in the map for us to try backing up

# Cached distribution
DIST = scipy_stats.norm(0, 1)

# Taken directly from 2021_G2
@functools.lru_cache()
def standard_ppf(conf: float) -> float:
    return DIST.ppf(conf)

# Taken directly from 2021_G2
def spread_points(current_point: Tuple[float, float], angles: np.array, distance: float, reverse: bool) -> np.array:
    curr_x, curr_y = current_point
    if reverse:
        angles = np.flip(angles)
    xs = np.cos(angles) * distance + curr_x
    ys = np.sin(angles) * distance + curr_y
    return np.column_stack((xs, ys))

# Based on function of same name from 2021_G2
def splash_zone(distance: float, angle: float, conf: float, skill: int, current_point: Tuple[float, float],
                is_sand: bool, is_sand_target: bool) -> np.array:
    conf_points = np.linspace(1 - conf, conf, 5)
    st_coeff = 2 if is_sand else 1
    distances = np.vectorize(standard_ppf)(conf_points) * st_coeff * (distance / skill) + distance
    angles = np.vectorize(standard_ppf)(conf_points) * st_coeff * (1/(2*skill)) + angle
    scale = 1.1

    if (distance <= 20 and not is_sand) or is_sand_target:
        scale = 1.0
    max_distance = distances[-1]*scale
    top_arc = spread_points(current_point, angles, max_distance, False)

    if distance > 20 or is_sand:
        min_distance = distances[0]
        bottom_arc = spread_points(current_point, angles, min_distance, True)
        return np.concatenate((top_arc, bottom_arc, np.array([top_arc[0]])))

    current_point = np.array([current_point])
    return np.concatenate((current_point, top_arc, current_point))
      

class Player(object):
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger, golf_map: sympy.Polygon, start: sympy.geometry.Point2D, target: sympy.geometry.Point2D, sand_traps, map_path: str, precomp_dir: str) -> None:
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

        # Initialize variables required for precomputing
        self.centroids_dict = {}
        self.point_in_sand_cache = {}

        # Check if the map has been precomputed
        precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))
        if os.path.isfile(precomp_path):
            with open(precomp_path, "rb") as f:
                self.shapely_map, self.shapely_sand_traps, self.all_sandtraps, self.centroids_dict = pickle.load(f)
        else:
            # If the map has not been precomputed, do so
            self.shapely_map = shapely.geometry.Polygon(golf_map.vertices)
            self.shapely_sand_traps = [shapely.geometry.Polygon(st.vertices) for st in sand_traps]
            self.all_sandtraps = shapely.ops.unary_union(self.shapely_sand_traps)
            self.centroids_dict = self.split_polygon(50)
            
            # Then dump the precomputation for the next run
            with open(precomp_path, 'wb') as f:
                pickle.dump([self.shapely_map, self.shapely_sand_traps, self.all_sandtraps, self.centroids_dict], f)
            
            # And save an image of the generated map
            regions_image_path = os.path.join(precomp_dir, "{}-regions.jpg".format(map_path))
            plt.figure(dpi=200)
            plt.axis('equal')
            plt.plot(*self.shapely_map.exterior.xy)
            plt.scatter([r[0] for r in self.centroids_dict.keys()], [r[1] for r in self.centroids_dict.keys()], color='red')
            for region in self.centroids_dict.values():
                plt.plot(*region['poly'].exterior.xy)
            plt.gca().invert_yaxis()
            plt.savefig(regions_image_path)
                
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.start = float(start.x), float(start.y)
        self.goal = float(target.x), float(target.y)

        self.np_map_points = None
        self.prev_rv = None

        # Record maxmimum distances and their potential distributions
        self.max_dist = 200 + self.skill
        self.max_ddist = scipy_stats.norm(self.max_dist, self.max_dist / self.skill)
        self.max_dist_st = self.max_dist/2
        self.max_ddist_st = scipy_stats.norm(self.max_dist_st, 2 * self.max_dist_st / self.skill)

        self.centroids = list(self.centroids_dict.keys())

        # Add start and end points to centroids and centroids_dict
        self.centroids_dict[self.start] = {'poly': None, 'is_in_sand': self.is_point_in_sand(self.start)}
        self.centroids_dict[self.goal] = {'poly': None, 'is_in_sand': self.is_point_in_sand(self.goal)}
        self.centroids.append(self.goal)

        # Confidence level
        self.confidence = HIGH_SKILL_CONFIDENCE
        if self.skill < SKILL_CONFIDENCE_THRESHHOLD:
            self.confidence = LOW_SKILL_CONFIDENCE

    # Taken directly from 2021_G2
    @functools.lru_cache()
    def _max_ddist_ppf(self, conf: float):
        return self.max_ddist.ppf(1.0 - conf)

    # Based on _max_ddist_ppf from 2021_G2
    @functools.lru_cache()
    def _max_ddist_st_ppf(self, conf: float):
        return self.max_ddist_st.ppf(1.0 - conf)
    
    # Based on function of same name from 2021_G2
    def splash_zone_within_map(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float) -> bool:
        distance = np.linalg.norm(np.array(current_point).astype(float) - np.array(target_point).astype(float))
        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(float(ty) - float(cy), float(tx) - float(cx))
        splash_zone_poly_points = splash_zone(float(distance), float(angle), float(conf), self.skill, current_point,
                                              self.is_point_in_sand(current_point),
                                              self.is_point_in_sand(target_point))
        return self.shapely_map.contains(ShapelyPolygon(splash_zone_poly_points))

    def pct_splash_zone_within_map(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float) -> float:
        distance = np.linalg.norm(np.array(current_point).astype(float) - np.array(target_point).astype(float))
        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(float(ty) - float(cy), float(tx) - float(cx))
        splash_zone_poly_points = splash_zone(float(distance), float(angle), float(conf), self.skill, current_point,
                                                self.is_point_in_sand(current_point),
                                                self.is_point_in_sand(target_point))
        splash_zone_shapely = ShapelyPolygon(splash_zone_poly_points)
        return splash_zone_shapely.intersection(self.shapely_map).area / splash_zone_shapely.area

    def pct_splash_zone_within_region(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float) -> float:
        distance = np.linalg.norm(np.array(current_point).astype(float) - np.array(target_point).astype(float))
        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(float(ty) - float(cy), float(tx) - float(cx))
        splash_zone_poly_points = splash_zone(float(distance), float(angle), float(conf), self.skill, current_point,
                                              self.is_point_in_sand(current_point),
                                              self.is_point_in_sand(target_point))

        corresp_region = self.centroids_dict[target_point]['poly']
        splash_zone_shapely = ShapelyPolygon(splash_zone_poly_points)

        if not corresp_region or not splash_zone_shapely.area: 
            return 0

        return splash_zone_shapely.intersection(corresp_region).area / splash_zone_shapely.area

    def create_vornoi_regions(self, poly: shapely.geometry.Polygon, region_num: int, point_spacing: float) -> List[shapely.geometry.Polygon]:
        points = []
        min_x, min_y, max_x, max_y = poly.bounds

        # Generate a dense grid of points within the bounds of a given map to produce homogenous regions
        # with centroids almost exactly in the middle when later clustering with kmeans
        for x in np.arange(min_x, max_x, point_spacing):
            for y in np.arange(min_y, max_y, point_spacing):
                pt = shapely.geometry.Point(x, y)
                if poly.contains(pt):
                    points.append([x, y])

        # Add random points to distribution to help clustering accuracy
        while len(points) < 100 * region_num:
            x, y = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            pt = shapely.geometry.Point(x, y)
            if poly.contains(pt):
                points.append([x, y])

        # Cluster the random points into groups using kmeans
        np_points = np.array(points)
        kmeans = Kmeans(np_points.shape[1], region_num)
        kmeans.train(np_points.astype(np.float32))

        # Generate a voronoi diagram from the centers of the generated regions
        center_points = shapely.geometry.MultiPoint(kmeans.centroids)
        regions = shapely.ops.voronoi_diagram(center_points, edges=False)

        # Intersect the generated regions with the given map
        regions = [r for r in [poly.intersection(region) for region in regions.geoms] if not r.is_empty]

        # Break possibly split regions into separate polygons (avoids tiny regions)
        flattened_regions = []
        for region in regions:
            if region.geom_type == 'MultiPolygon':
                flattened_regions.extend(list(region.geoms))
            elif region.geom_type == 'Polygon':
                flattened_regions.append(region)

        # Remove regions with small perimiters with respect to their area
        flattened_regions = [r for r in flattened_regions if r.length/(r.area**(1/2)) < 15]

        return flattened_regions

    def split_polygon(self, region_num: int) -> Dict[Tuple[float, float], Dict[str, any]]:
        """ Split a given Golf Map into regions of roughly equal size.
        Based on an algorithm described by Paul Ramsey: http://blog.cleverelephant.ca/2018/06/polygon-splitting.html

        Args:
            golf_map (shapely.geometry.Polygon): The Golf Map to split into equal sized regions
            sand_traps (shapely.geometry.Polygon): A list of Sand Traps contained within the Golf Map 
            regions (int): The number of roughly equal sized regions to split the map into

        Returns:
            Dict[Tuple[float, float], Dict[str, any]]  Returns a dict with region centroid x/y tuples as keys and dicts as values
        """

        golf_map_with_holes = shapely.geometry.Polygon(self.shapely_map.exterior.coords, [list(st.exterior.coords) for st in self.shapely_sand_traps])

        grass_regions = int(golf_map_with_holes.area/self.shapely_map.area * region_num)
        regions = self.create_vornoi_regions(golf_map_with_holes, grass_regions, POINT_SPACING)

        # Find total and avg area
        avg_area_centroid = golf_map_with_holes.area/len(regions)

        st_regions = []
        for st in self.shapely_sand_traps:
            num_points = max(floor(st.area/avg_area_centroid), 1)
        # If there are 1 more or points run k means else use already exsisting geometry
            if num_points > 1:
                st_regions.extend(self.create_vornoi_regions(st, num_points, POINT_SPACING))
            else:
                st_regions.append(st)

        # add all regions together and prepare returnables
        regions.extend(st_regions)

        region_centers = [tuple(polylabel([region.exterior.coords])) for region in regions]


        centroids_dict = {
            region_center: {'poly': region, 'is_in_sand': self.is_point_in_sand(region_center)} 
            for region_center, region in zip(region_centers, regions)
        }

        return centroids_dict

    def numpy_adjacent_and_dist(self, point: Tuple[float, float], conf: float):
        shooting_from_st = self.is_point_in_sand(point)

        max_distance_no_roll = self._max_ddist_ppf(conf) if not shooting_from_st else self._max_ddist_st_ppf(conf)
        max_distance_with_grass_roll = max_distance_no_roll * 1.1
        max_distance_with_st_roll = max_distance_no_roll + 0.01

        point_distances = cdist(self.np_map_points, np.array([np.array(point)]), 'euclidean')
        point_distances = point_distances.flatten()

        reachable_points = []
        goal_distances = []
        st_points = []
        for pt, distance_to_pt, distance_to_goal in zip(self.np_map_points, point_distances, self.np_goal_dist):
            point_in_st = self.is_point_in_sand(point)

            if point_in_st:
                st_points.append(pt)

            if distance_to_pt <= (max_distance_with_st_roll if point_in_st else max_distance_with_grass_roll):
                reachable_points.append(pt)
                goal_distances.append(distance_to_goal)

        return reachable_points, goal_distances, st_points

    # Based on function of same name from 2021_G2, but heavily modified
    def next_target(self, currrent_point: Tuple[float, float], goal: Point2D, conf: float) -> Union[None, Tuple[float, float]]:
        goal_point = float(goal.x), float(goal.y)
        heap = [self.ScoredPoint(self, currrent_point, goal_point, actual_cost=0.0)]
        start_point = heap[0].point

        # Lower our confidence if we are putting so ensure we take more careful shots
        if heap[0].goal_dist <= 20:
            self.confidence = PUTTING_CONFIDENCE

        # Cache best cost to avoid adding worse points to the heap
        best_cost = {tuple(currrent_point): 0.0}
        visited = set()
        points_checked = 0
        while len(heap) > 0:
            next_sp = heapq.heappop(heap)
            next_p = next_sp.point

            if next_p in visited:
                continue
            if next_sp.actual_cost > 10:
                continue
            if next_sp.actual_cost > 0 and not self.splash_zone_within_map(next_sp.previous.point, next_p, conf):
                if next_p in best_cost:
                    del best_cost[next_p]
                continue
            visited.add(next_p)

            if np.linalg.norm(np.array(self.goal) - np.array(next_p)) <= 5.4 / 100.0:
                # We can reach the hole, so all we care about is the next point
                while next_sp.previous.point != start_point:
                    next_sp = next_sp.previous
                return next_sp.point
            
            # Add adjacent points to heap
            reachable_points, goal_dists, _ = self.numpy_adjacent_and_dist(next_p, conf)
           
            for i in range(len(reachable_points)):
                candidate_point = tuple(reachable_points[i])
                goal_dist = goal_dists[i]
                new_point = Player.ScoredPoint(self, candidate_point, goal_point, actual_cost=next_sp.actual_cost + 1, previous=next_sp, goal_dist=goal_dist, skill=self.skill)
                
                # Check to see how much of the splash zone is in sand traps and adjust point weight accordingly
                pct_splash_zone_not_in_target_region = 1 - self.pct_splash_zone_within_region(next_sp.point, new_point.point, conf) 
                if pct_splash_zone_not_in_target_region > 0.1 and pct_splash_zone_not_in_target_region < 1:
                    new_point.f_cost = new_point.f_cost + (pct_splash_zone_not_in_target_region * SAND_TRAP_IN_SPLASH_ZONE_WEIGHT)
                
                if candidate_point not in best_cost or best_cost[candidate_point] > new_point.f_cost:
                    points_checked += 1
                    best_cost[new_point.point] = new_point.f_cost
                    heapq.heappush(heap, new_point)

        # No path found
        return None

    def _initialize_map_points(self, goal: Tuple[float, float], golf_map: Polygon, sand_traps):
        # Storing the points as numpy array
        np_map_points = []
        pp = self.centroids

        for point in pp:
            x, y = point
            np_map_points.append(np.array([x, y]))

        self.np_map_points = np.array(np_map_points)
        self.np_goal_dist = cdist(self.np_map_points, np.array([np.array(self.goal)]), 'euclidean')
        self.np_goal_dist = self.np_goal_dist.flatten()

    def is_point_in_sand(self, point: Tuple[float, float])-> bool:
        """Helper function to check whether a given point is within a sandtrap"""

        if point in self.centroids_dict:
            return self.centroids_dict[point]['is_in_sand']
        elif point in self.point_in_sand_cache:
            return self.point_in_sand_cache[point]

        # Cache each point we check to speed future checks
        is_in_sand = self.all_sandtraps.contains(shapely.geometry.Point(point))
        self.point_in_sand_cache[point] = is_in_sand

        return is_in_sand

    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, sand_traps, curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D, prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
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
            self._initialize_map_points((gx, gy), golf_map, sand_traps)

        # Optimization to retry missed shots
        if self.prev_rv is not None and curr_loc == prev_loc:
            return self.prev_rv

        target_point = None
        confidence = self.confidence
        cl = float(curr_loc.x), float(curr_loc.y)

        print('\n')
        print(f"===== SHOT NUMBER {score} =====")
        
        while target_point is None:
            if confidence <= 0.0:
                return None

            target_point = self.next_target(cl, target, confidence)
            if target_point:
                continue

            confidence -= 0.05

        # Adjust shot so we stop exactly on the target, if possible (accounting for roll)
        original_target = target_point
        current_point = np.array(tuple(curr_loc)).astype(float)
        original_dist = np.linalg.norm(np.array(target_point) - current_point)
        v = np.array(target_point) - current_point # Vector from current to target
        u = v / original_dist # Unit vector pointing from current to target

        print(f"Current location: {round(cl[0], 2)}, {round(cl[1], 2)}")
        print(F"Target point: {round(original_target[0], 2)}, {round(original_target[1], 2)}")

        # Back up target to account for roll
        if original_dist >= 20.0:
            max_roll = (original_dist / 1.1) * 0.1
            # If target is goal, try to overshoot by 1 meter to roll into the hole
            max_offset = max_roll if tuple(target_point) != self.goal else (max_roll - METERS_TO_OVERSHOOT_HOLE)
            offset = 0
            dist = original_dist
            
            # If target point is in a sand trap, try backing up, up to X% more, to avoid
            original_target_in_st = self.is_point_in_sand(target_point)
            if original_target_in_st:
                print('Target point is in a sand trap')
                max_offset = max_offset * (1 + EXTRA_BACKUP_PCT_TO_AVOID_SAND_TRAPS)
            
            # Keep track of if we have backed into or started in grass so we don't go too far and hit sand
            hit_grass = not original_target_in_st

            while offset <= max_offset and self.pct_splash_zone_within_map(tuple(current_point), tuple(target_point), confidence) > PCT_SPLASH_ZONE_IN_MAP_TO_BACK_UP:
                offset += BACKUP_INCREMENT
                dist = original_dist - min(offset, max_offset)
                new_target_point = current_point + u * dist
                                
                new_target_in_st = self.is_point_in_sand(tuple(target_point))
                if not hit_grass and not new_target_in_st:
                    hit_grass = True
                backed_into_sand = hit_grass and new_target_in_st

                if backed_into_sand:
                    print("Backed up into sand...undoing the backup")
                    offset -= BACKUP_INCREMENT * UNDO_BACKUP_INCREMENTS
                    dist = original_dist - max(offset, 0)
                    target_point = current_point + u * dist
                    break

                target_point = new_target_point

            print(f"Offset: {round(original_dist - dist, 2)} (max allowed offset: {round(max_offset, 2)})")

        # If target is goal and we are putting, overshoot by 10% so we roll into the hole
        elif tuple(target_point) == self.goal:
            dist = original_dist + METERS_TO_OVERSHOOT_HOLE
            target_point = current_point + u * dist

        if tuple(target_point) != original_target:    
            print(f"Corrected target point: {round(target_point[0], 2)}, {round(target_point[1], 2)}")

        print('\n')

        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(ty - cy, tx - cx)
        distance = curr_loc.distance(Point2D(target_point, evaluate=False))
        distance = min(distance, 200 + self.skill)

        rv = distance, angle
        self.prev_rv = rv
        return rv

    # Based on function of same name from 2021_G2
    class ScoredPoint(object):
        """Scored point class for use in A* search algorithm"""
        def __init__(self, player, point: Tuple[float, float], goal: Tuple[float, float], actual_cost=float('inf'), previous=None, goal_dist=None, skill=50, conf=0.95):
            self.point = point
            self.goal = goal
            self.goal_dist = goal_dist
            self.previous = previous
            self.actual_cost = actual_cost

            if self.goal_dist is None:
                a = np.array(self.point)
                b = np.array(self.goal)
                self.goal_dist = np.linalg.norm(a - b)

            sandtrap_cost = FIXED_SANDTRAP_COST if player.is_point_in_sand(point) else 0  # sandtrap adds an extra .5 shots
            max_dist = (200 + skill) * 1.1
            
            reachable_points, _, st_points = player.numpy_adjacent_and_dist(point, conf)

            reachable_pts_ct = len(reachable_points)
            reachable_st_pts_ct = len(st_points)
            reachable_grass_pts = reachable_pts_ct - reachable_st_pts_ct
            reachable_points_cost = (reachable_grass_pts * 1 + reachable_st_pts_ct * (1 + FIXED_SANDTRAP_COST)) / reachable_pts_ct

            self.h_cost = \
                (self.goal_dist / max_dist) * REMAINING_SHOTS_WEIGHT  \
                + sandtrap_cost * SAND_TRAP_WEIGHT \
                + reachable_points_cost * REACHABLE_POINTS_WEIGHT

            self.f_cost = self.actual_cost + self.h_cost
        
        def __lt__(self, other):
            return self.f_cost < other.f_cost

        def __eq__(self, other):
            return self.point == other.point
        
        def __hash__(self):
            return hash(self.point)
        
        def __repr__(self):
            return f"ScoredPoint(point = {self.point}, h_cost = {self.h_cost})"