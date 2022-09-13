import math
import random

import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
from shapely.geometry import Polygon, Point, LineString
from time import time
from shapely.ops import triangulate
import math


def line_in_polygon(point_a, point_b, polygon):
    segment_land = LineString([point_a, point_b])
    if_encloses = polygon.contains(segment_land)
    return if_encloses


def get_roll_final_point(point_a, distance, angle):
    final_point = Point(
        point_a.x + (1.1) * distance * np.cos(angle),
        point_a.y + (1.1) * distance * np.sin(angle))
    return final_point


def is_roll_in_polygon(point_a, distance, angle, polygon):
    curr_point = Point(point_a.x + distance * np.cos(angle),
                       point_a.y + distance * np.sin(angle))

    final_point = get_roll_final_point(point_a, distance, angle)
    start = time()
    if_encloses = line_in_polygon(curr_point, final_point, polygon)
    end = time()
    # print("line-time", end - start)
    #
    # start = time()
    # curr_point.within(polygon)
    # final_point.within(polygon)
    # end = time()
    # print("points", end-start)
    return if_encloses


def convert_sympy_shapely(point):
    return Point(point.x, point.y)


def predict_num_shots(distance_to_hole, skill):
    max_distance = 200 + skill
    distance_groups = [20, 100] + ([(200 + skill - 20) * i for i in range(10)])
    for i in range(len(distance_groups)):
        if int(distance_to_hole) <= distance_groups[i]:
            dg = i
            break
    if dg == 0:
        next_min = 0
    else:
        next_min = distance_groups[dg-1]
    decimal = (distance_to_hole - next_min) / (distance_groups[dg] - next_min)
    dg += decimal
    dg+=1
    return dg


def score_paths(start_point, target, paths_to_search, polygon, skill, rng,if_print=False):
    if len(paths_to_search) == 0:
        return None

    start_to_hole_distance = start_point.distance(target)
    all_h = np.array([path.heuristic(start_to_hole_distance, skill) for path in paths_to_search])
    all_c = np.array([path.confidence(polygon, skill, rng) for path in paths_to_search])

    best_i = np.argmin(all_h + all_c)
    return paths_to_search[best_i]


def direct_distance_angle(curr_loc, target, skill):
    required_dist = curr_loc.distance(target)
    roll_factor = 1.1
    if required_dist < 20:
        roll_factor = 1.0
    max_dist_traveled = 200 + skill
    distance = min(max_dist_traveled, (required_dist / roll_factor))
    angle = math.atan2(target.y - curr_loc.y, target.x - curr_loc.x)

    return distance, angle


def generate_points(curr_loc, target, polygon, skill, increment=25):
    distance, angle = direct_distance_angle(curr_loc, target, skill)

    if distance < 20 and line_in_polygon(curr_loc, target, polygon):
        return [LandingPoint(target, distance, angle, curr_loc, target)]

    target_to_aim = Point(curr_loc.x + distance * np.cos(angle), curr_loc.y + distance * np.sin(angle))
    points = [LandingPoint(target_to_aim, distance, angle, curr_loc, target)]
    r = min(distance + 70, 200 + skill)
    while r > 0:
        semicircle_length = np.pi * r
        num_sector = int(semicircle_length / increment)  # divide the semicircle into equally sized sectors
        num_sector = num_sector if num_sector % 2 == 0 else num_sector + 1
        if num_sector == 0:
            r -= increment
            continue
        arc_length = semicircle_length / num_sector
        angle_increment = np.pi / (2 * num_sector)
        for i in range(0, int(num_sector) + 1):
            new_angle = float(angle + (i * angle_increment))
            point = Point(curr_loc.x + r * np.cos(new_angle),
                          curr_loc.y + r * np.sin(new_angle))

            if LandingPoint.is_on_land(point, polygon):
                lp = LandingPoint(point, r, new_angle, curr_loc, target)
                points.append(lp)
            if i > 0:
                new_angle = float(angle - (i * angle_increment))
                point = Point(curr_loc.x + r * np.cos(new_angle),
                              curr_loc.y + r * np.sin(new_angle))

                if LandingPoint.is_on_land(point, polygon):
                    lp = LandingPoint(point, r, new_angle, curr_loc, target)
                    points.append(lp)
        r -= increment

    return points


class LandingPoint(object):
    def __init__(self, point, distance_from_origin, angle_from_origin, start_point, hole_point):
        # distance_from_origin is the distance from our curr_location to the landing point

        self.point = point
        self.distance_from_origin = distance_from_origin
        self.angle_from_origin = angle_from_origin
        self.hole = hole_point
        self.score_threshold = 95
        self.trials = 12
        self.start_point = start_point


    @staticmethod
    def is_on_land(point, polygon):
        return point.within(polygon)

    def confidence(self, polygon, skill, rng):
        if hasattr(self, 'shot_confidence'):
            return self.shot_confidence
        else:
            intended_distance = self.distance_from_origin
            intended_angle = self.angle_from_origin
            successful = 0
            for t in range(0, self.trials):
                actual_distance = rng.normal(intended_distance, intended_distance / skill)
                actual_angle = rng.normal(intended_angle, 1 / (2 * skill))
                if is_roll_in_polygon(self.start_point, actual_distance, actual_angle, polygon):
                    successful += 1
            frac = (successful / self.trials)
            frac = frac if frac != 0 else 0.05
            self.successful_trials = successful
            self.shot_confidence = 1 / frac
            return self.shot_confidence

    def heuristic(self, initial_distance, skill):
        final_point = get_roll_final_point(self.start_point, self.distance_from_origin, self.angle_from_origin)
        distance_to_hole = final_point.distance(self.hole)
        # distance_to_hole = 1 if distance_to_hole < 1 else distance_to_hole
        # # self.normalized_hueristic = (initial_distance - distance_to_hole) / (initial_distance)
        # self.h = (200 + skill) / distance_to_hole
        self.h = predict_num_shots(distance_to_hole, skill)
        return self.h


class MultipleLandingPoints:

    def __init__(self, start_lp):
        self.path = [start_lp]

    def heuristic(self, initial_distance, skill):
        return self.path[-1].heuristic(initial_distance, skill)

    def confidence(self, polygon, skill, rng):
        total = 0
        for lp in self.path:
            total += lp.confidence(polygon, skill, rng)
        return total

    def score(self, polygon, skill, rng, initial_distance):
        val = (self.heuristic(initial_distance) + (self.confidence(polygon, skill, rng)))
        return val/len(self.path)

    def add_point(self, polygon, skill, rng):
        last_point = self.path[-1]

        landing_points = generate_points(last_point.point, last_point.hole, polygon, skill)
        next_point = score_paths(last_point.point, last_point.hole, landing_points, polygon, skill, rng)
        if next_point:
            self.path.append(next_point)

    def distance_to_hole(self):
        last_point = self.path[-1]
        return last_point.point.distance(last_point.hole)


class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger, golf_map: sympy.Polygon, start: sympy.geometry.Point2D, target: sympy.geometry.Point2D, map_path: str, precomp_dir: str) -> None:
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


    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D,
             curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D,
             prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
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

        if score == 1:
            self.shapely_polygon = Polygon([(p.x, p.y) for p in golf_map.vertices])

        curr_loc = convert_sympy_shapely(curr_loc)
        target = convert_sympy_shapely(target)
        landing_points = generate_points(curr_loc, target, self.shapely_polygon, self.skill)

        paths = [MultipleLandingPoints(lp) for lp in landing_points]
        for path in paths:
            if path.distance_to_hole() > 20:
                path.add_point(self.shapely_polygon, self.skill, self.rng)
            if path.distance_to_hole()> 20:
                path.add_point(self.shapely_polygon, self.skill, self.rng)

        largest_point = score_paths(curr_loc, target, paths, self.shapely_polygon, self.skill, self.rng, if_print=True)
        return largest_point.path[0].distance_from_origin, largest_point.path[0].angle_from_origin