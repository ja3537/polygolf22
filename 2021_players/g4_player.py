import sys

import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
import constants

from shapely import geometry
from collections import defaultdict
import shapely.geometry
import math


def get_distance(point1, point2):
    return math.sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2))


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger, golf_map: sympy.Polygon,
                 start: sympy.geometry.Point2D, target: sympy.geometry.Point2D, map_path: str,
                 precomp_dir: str) -> None:
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
        precomp_path = os.path.join(precomp_dir, "{}_skill-{}.pkl".format(map_path, skill))
        # # if doesn't depend on skill
        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))
        self.point_dict = defaultdict(float)
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.risk = 0.1
        self.simulate_times = 100
        self.tolerant_times = self.simulate_times * self.risk

        # self.pre_compute_simulate_times = 10
        # self.pre_compute_risk_rate = 0.2
        # self.pre_compute_tolerant_times = self.pre_compute_simulate_times * self.pre_compute_risk_rate

        self.allowed_distance = constants.max_dist + self.skill

        # # precompute check
        if os.path.isfile(precomp_path):
            # Getting back the objects:
            with open(precomp_path, "rb") as f:
                grid_scores, point_map = pickle.load(f)
        else:
            # Compute objects to store
            grid_scores, point_map = self.make_grid(golf_map, target)

            # Dump the objects
            with open(precomp_path, 'wb') as f:
                pickle.dump([grid_scores, point_map], f)

        for x_index in range(len(point_map)):
            for y_index in range(len(point_map[0])):
                if int(grid_scores[x_index][y_index]) != 100:
                    self.point_dict[Point(point_map[x_index][y_index].x, point_map[x_index][y_index].y)] = \
                    grid_scores[x_index][y_index]

        self.point_dict[Point(target.x, target.y)] = 0

        self.turn = 0
        self.shapely_golf_map = None

        self.last_result = None
        self.prev_sample_point = []

        self.prev_expected_scores = defaultdict(int)
        self.max_same_score_times = 1

        curr_expected_score = self.get_expected_score(self.point_dict, start)
        self.prev_expected_scores[curr_expected_score] += 1

    def water_boolean(self, poly, grid_points):
        water_grid = []

        for i, row in enumerate(grid_points):
            water_grid.append([])
            for j, point in enumerate(row):
                thebool = poly.contains(point)

                water_grid[i].append(thebool)

        return np.array(water_grid)

    def make_grid(self, golf_map: sympy.Polygon, target: sympy.geometry.Point2D):
        poly = geometry.Polygon([p.x, p.y] for p in golf_map.vertices)
        target_shapely = geometry.Point(target[0], target[1])

        (xmin, ymin, xmax, ymax) = golf_map.bounds
        list_of_lists = []
        list_of_distances = []

        dimension = 60
        grid_of_scores = np.array(np.ones((dimension, dimension)) * 100)

        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)

        xcoords, ycoords = np.meshgrid(np.linspace(xmin, xmax, dimension), np.linspace(ymin, ymax, dimension))

        for x_index in range(len(xcoords)):
            list_of_lists.append([])
            list_of_distances.append([])
            for y_index in range(len(ycoords)):
                # considered_point = sympy.geometry.Point2D(xcoords[y_index,x_index],ycoords[y_index,x_index])
                considered_point = geometry.Point(xcoords[y_index, x_index], ycoords[y_index, x_index])
                list_of_lists[x_index].append(considered_point)

        grid_of_scores = self.real_bfs(poly, list_of_lists, target_shapely, self.allowed_distance, grid_of_scores)

        return grid_of_scores, list_of_lists

    def real_bfs(self, poly, list_of_lists, target_shapely, allowed_distance, grid_of_scores):
        queue = []
        water_grid = self.water_boolean(poly, list_of_lists)  # True if on LAND
        for x_index in range(len(list_of_lists)):
            for y_index in range(len(list_of_lists[0])):
                if not water_grid[x_index][y_index]:
                    continue

                thedistance = target_shapely.distance(list_of_lists[x_index][y_index])
                if (thedistance < allowed_distance):
                    #     is_safe = True
                    #     failed_times = 0
                    #     angle = sympy.atan2(target_shapely.y - list_of_lists[x_index][y_index].y, target_shapely.x - list_of_lists[x_index][y_index].x)
                    #     # simulate the actual situation to ensure fail times will not be larger than self.tolerant_times
                    #     for _ in range(self.pre_compute_simulate_times):
                    #
                    #         is_succ, _ = self.simulate_shapely_once(thedistance, angle, list_of_lists[x_index][y_index], poly, True)
                    #         if not is_succ:
                    #             failed_times += 1
                    #             if failed_times > self.pre_compute_tolerant_times:
                    #                 is_safe = False
                    #                 break
                    #
                    #     if not is_safe:
                    #         continue
                    queue.append((x_index, y_index))
                    if thedistance < constants.min_putter_dist:
                        grid_of_scores[x_index][y_index] = 0.5
                    else:
                        grid_of_scores[x_index][y_index] = 1

        y_cell_size = list_of_lists[0][1].y - list_of_lists[0][0].y
        x_cell_size = list_of_lists[1][0].x - list_of_lists[0][0].x
        x_cell_range = int(allowed_distance / x_cell_size) + 1
        y_cell_range = int(allowed_distance / y_cell_size) + 1

        while (len(queue) != 0):

            elem = queue.pop()
            # get all points that are < distance from elem
            elem_score = grid_of_scores[elem[0]][elem[1]]

            x_index_start = 0 if elem[0] - x_cell_range < 0 else elem[0] - x_cell_range
            x_index_end = len(list_of_lists) if elem[0] + x_cell_range + 1 > len(list_of_lists) else elem[
                                                                                                         0] + x_cell_range + 1
            y_index_start = 0 if elem[1] - y_cell_range < 0 else elem[1] - y_cell_range
            y_index_end = len(list_of_lists[0]) if elem[1] + y_cell_range + 1 > len(list_of_lists[0]) else elem[
                                                                                                               1] + y_cell_range + 1
            '''
            x_index_start = 0
            x_index_end = len(list_of_lists)
            y_index_start = 0
            y_index_end = len(list_of_lists[0])
            '''
            for x_index in range(x_index_start, x_index_end):
                for y_index in range(y_index_start, y_index_end):
                    elem_point = list_of_lists[elem[0]][elem[1]]
                    if not water_grid[x_index, y_index] or elem_score + 1 >= grid_of_scores[x_index][y_index]:
                        continue

                    distance = elem_point.distance(list_of_lists[x_index][y_index])
                    if distance <= allowed_distance:
                        # is_safe = True
                        # failed_times = 0
                        # angle = sympy.atan2(list_of_lists[x_index][y_index].y - elem_point.y,
                        #                     list_of_lists[x_index][y_index].x - elem_point.x)
                        # # simulate the actual situation to ensure fail times will not be larger than self.tolerant_times
                        # for _ in range(self.pre_compute_simulate_times):
                        #
                        #     is_succ, _ = self.simulate_shapely_once(distance, angle, elem_point,
                        #                                             poly, True)
                        #     if not is_succ:
                        #         failed_times += 1
                        #         if failed_times > self.pre_compute_tolerant_times:
                        #             is_safe = False
                        #             break
                        #
                        # if not is_safe:
                        #     continue

                        grid_of_scores[x_index][y_index] = elem_score + 1
                        queue.append((x_index, y_index))
                        # points_to_consider.append((x_index, y_index))

            # for point in points_to_consider:
            #     if water_grid[point[0], point[1]]:
            #         (x_index, y_index) = point
            #         if elem_score + 1 < grid_of_scores[x_index][y_index]:
            #             grid_of_scores[x_index][y_index] = elem_score + 1
            #             queue.append(point)
        return grid_of_scores

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

        if self.turn == 0:
            self.shapely_golf_map = shapely.geometry.polygon.Polygon(golf_map.vertices)

        self.turn += 1

        if not prev_admissible and self.last_result is not None:
            distance, angle = self.last_result
            final_point = self.get_expected_next_point(distance, angle, curr_loc)
            self.prev_expected_scores[self.get_expected_score(self.point_dict, final_point)] += 1
            return self.last_result

        # 1. always try greedy first
        required_dist = curr_loc.distance(target)
        roll_factor = 1. + constants.extra_roll
        if required_dist < constants.min_putter_dist:
            roll_factor = 1.0
        distance = sympy.Min(constants.max_dist + self.skill, required_dist / roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)

        can_try_greedy = True
        if required_dist > constants.max_dist + self.skill:
            # if the distance > largest distance
            # we can always try the greedy first, as long as the score of that landing point is decreasing
            curr_expected_score = self.get_expected_score(self.point_dict, curr_loc)
            final_point = self.get_expected_next_point(distance, angle, curr_loc)

            greedy_expected_score = self.get_expected_score(self.point_dict, final_point)
            if curr_expected_score <= greedy_expected_score:
                can_try_greedy = False

        if can_try_greedy:
            is_greedy = True
            failed_times = 0
            # simulate the actual situation to ensure fail times will not be larger than self.tolerant_times
            for _ in range(self.simulate_times):
                is_succ, _ = self.simulate_shapely_once(distance, angle, curr_loc, self.shapely_golf_map)
                if not is_succ:
                    failed_times += 1
                    if failed_times > self.tolerant_times:
                        is_greedy = False
                        break
            if is_greedy:
                final_point = self.get_expected_next_point(distance, angle, curr_loc)
                self.prev_expected_scores[self.get_expected_score(self.point_dict, final_point)] += 1
                self.logger.info(str(self.turn) + "select greedy strategy to go")
                self.last_result = (distance, angle)
                return (distance, angle)

        self.logger.info(str(self.turn) + "sample points to go")
        desire_distance, desire_angle = self.get_safe_sample_points_inside_circle(self.point_dict, curr_loc, distance,
                                                                                  target, golf_map, prev_admissible)
        self.last_result = (desire_distance, desire_angle)
        final_point = self.get_expected_next_point(desire_distance, desire_angle, curr_loc)
        self.prev_expected_scores[self.get_expected_score(self.point_dict, final_point)] += 1

        return (desire_distance, desire_angle)

    def simulate_shapely_once(self, distance, angle, curr_loc, golf_map, is_simple=False):
        actual_distance = self.rng.normal(distance, distance / self.skill)
        actual_angle = self.rng.normal(angle, 1 / (2 * self.skill))

        # landing_point means the land point(the golf can skip for a little distance),
        # final_point means the final stopped point, it is not equal
        if distance < constants.min_putter_dist:
            landing_point = shapely.geometry.Point(curr_loc.x, curr_loc.y)
            final_point = shapely.geometry.Point(curr_loc.x + actual_distance * np.cos(actual_angle),
                                                 curr_loc.y + actual_distance * np.sin(actual_angle))

        else:
            landing_point = shapely.geometry.Point(curr_loc.x + actual_distance * np.cos(actual_angle),
                                                   curr_loc.y + actual_distance * np.sin(actual_angle))
            final_point = shapely.geometry.Point(
                curr_loc.x + (1. + constants.extra_roll) * actual_distance * np.cos(actual_angle),
                curr_loc.y + (1. + constants.extra_roll) * actual_distance * np.sin(actual_angle))

        is_inside = golf_map.contains(landing_point) and golf_map.contains(final_point)

        if not is_simple and is_inside:
            test_times = 10
            deltax = (final_point.x - landing_point.x) / test_times
            deltay = (final_point.y - landing_point.y) / test_times
            for i in range(1, test_times - 1):
                new_point = shapely.geometry.Point(landing_point.x + i * deltax, landing_point.y + i * deltay)
                if not golf_map.contains(new_point):
                    is_inside = False
                    break
        return is_inside, final_point

    def diff_prev_sample_point(self, point):
        if len(self.prev_sample_point) == 0:
            return True

        if point in self.prev_sample_point:
            return False

        return True

    # points_score --> dictionary of (point, score)
    def get_safe_sample_points_inside_circle(self, points_score, curr_loc, radius, target, golf_map, prev_admissible):
        circle_points = dict()
        for point in points_score.keys():
            if get_distance(curr_loc, point) <= self.allowed_distance:
                circle_points[point] = points_score[point]

        sorted_points_score = dict(sorted(circle_points.items(), key=lambda x: x[1]))
        curr_expected_score = self.get_expected_score(self.point_dict, curr_loc)
        if len(self.prev_expected_scores) > 0:
            curr_expected_score = min(curr_expected_score, min(self.prev_expected_scores.keys()))

        if len(sorted_points_score) == 0:
            smallest_score = curr_expected_score
        else:
            smallest_score = min(sorted_points_score.values())

        middle_point_max_succ_times = -1
        now_max_succ_times = -1
        prev_sample_point_1 = None
        prev_sample_point_2 = None
        desire_distance_1, desire_angle_1 = None, None
        if curr_expected_score > smallest_score:
            self.logger.info(str(self.turn) + "max_possible_score < curr_expected_score")
            max_possible_score = curr_expected_score - 0.5

            smallest_score_points = dict()
            for point, value in sorted_points_score.items():
                if value <= max_possible_score:
                    smallest_score_points[point] = (value, get_distance(target, point))

            closest2target_points = dict(sorted(smallest_score_points.items(), key=lambda x: (x[1][0], x[1][1])))

            safe_point = None
            unsafe_points2score = dict()
            for point in closest2target_points.keys():
                if not self.diff_prev_sample_point(point):
                    continue

                succ_times = 0
                for _ in range(self.simulate_times):
                    angle = sympy.atan2(point.y - curr_loc.y, point.x - curr_loc.x)
                    is_succ, _ = self.simulate_shapely_once(get_distance(curr_loc, point), angle, curr_loc,
                                                            self.shapely_golf_map)
                    succ_times += is_succ

                if succ_times / self.simulate_times >= 1 - self.risk:
                    safe_point = point
                    break

                unsafe_points2score[point] = succ_times

            if safe_point:
                self.prev_sample_point.append(safe_point)
                desire_distance = get_distance(curr_loc, safe_point)
                desire_angle = sympy.atan2(safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
                return (desire_distance, desire_angle)

            unsafe_points = sorted(unsafe_points2score.items(), key=lambda x: -x[1])
            safe_point = unsafe_points[0][0]
            safe_point_max_succ_times = unsafe_points[0][1]
            if safe_point_max_succ_times == 0:
                quadrant = None
            else:
                quadrant = (safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
            desire_angle = sympy.atan2(safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
            self.logger.info(str(self.turn) + "reach unsafe state")

            # try to go to sample around middle points
            desire_distance, desire_angle, middle_point_max_succ_times = self.go_for_middle_points_in_circle(curr_loc,
                                                                                                             radius,
                                                                                                             golf_map,
                                                                                                             quadrant,
                                                                                                             desire_angle,
                                                                                                             curr_expected_score,
                                                                                                             target)
            if safe_point_max_succ_times >= middle_point_max_succ_times:
                self.logger.info(str(self.turn) + "still choose sample points")
                prev_sample_point_1 = safe_point
                desire_distance = get_distance(curr_loc, safe_point)
                desire_angle = sympy.atan2(safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
            elif middle_point_max_succ_times != 0:
                self.logger.info(str(self.turn) + "choose new sample points around the middle points")
            # sample_safe_point, sample_safe_point_max_succ_times = self.go_for_sample_points_in_circle(curr_loc, radius,
            #                                                                                           quadrant,
            #                                                                                           desire_angle,
            #                                                                                           max_possible_score)
            # if sample_safe_point_max_succ_times is None or safe_point_max_succ_times > sample_safe_point_max_succ_times:
            #     self.logger.info(str(self.turn) + "still choose sample points")
            #     desire_distance = get_distance(curr_loc, safe_point)
            #     desire_angle = sympy.atan2(safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
            # else:
            #     self.logger.info(str(self.turn) + "choose new sample points")
            #     desire_distance = get_distance(curr_loc, sample_safe_point)
            #     desire_angle = sympy.atan2(sample_safe_point.y - curr_loc.y, sample_safe_point.x - curr_loc.x)

            now_max_succ_times = max(safe_point_max_succ_times, middle_point_max_succ_times)
            if now_max_succ_times / self.simulate_times > 1 - self.risk * 2 or \
                    (now_max_succ_times / self.simulate_times >= 0.5 and
                     self.prev_expected_scores[curr_expected_score] >= self.max_same_score_times):
                if prev_sample_point_1:
                    self.prev_sample_point.append(prev_sample_point_1)
                return (desire_distance, desire_angle)

            desire_distance_1, desire_angle_1 = desire_distance, desire_angle

        self.logger.info(str(self.turn) + "max_possible_score = curr_expected_score")
        max_possible_score = curr_expected_score
        smallest_score_points = dict()
        for point, value in sorted_points_score.items():
            if value == max_possible_score:
                smallest_score_points[point] = (value, get_distance(target, point))

        curr2now_points = dict(sorted(smallest_score_points.items(), key=lambda x: (x[1][0], x[1][1])))

        safe_point_2 = None
        unsafe_points2score = dict()
        for point in curr2now_points.keys():
            if not self.diff_prev_sample_point(point):
                continue
            succ_times = 0
            for _ in range(self.simulate_times):
                angle = sympy.atan2(point.y - curr_loc.y, point.x - curr_loc.x)
                is_succ, _ = self.simulate_shapely_once(get_distance(curr_loc, point), angle, curr_loc,
                                                        self.shapely_golf_map)
                succ_times += is_succ

            if succ_times / self.simulate_times >= 1 - self.risk:
                safe_point_2 = point
                break

            unsafe_points2score[point] = succ_times

        if safe_point_2:
            self.prev_sample_point.append(safe_point_2)
            desire_distance_2 = get_distance(curr_loc, safe_point_2)
            desire_angle_2 = sympy.atan2(safe_point_2.y - curr_loc.y, safe_point_2.x - curr_loc.x)
            return (desire_distance_2, desire_angle_2)

        self.logger.info(str(self.turn) + "reach unsafe state")
        if len(unsafe_points2score) == 0:
            safe_point_max_succ_times_2 = 0
            quadrant = None
            desire_angle_2 = None
            desire_distance_2 = None
        else:
            unsafe_points = sorted(unsafe_points2score.items(), key=lambda x: -x[1])
            safe_point_2 = unsafe_points[0][0]
            safe_point_max_succ_times_2 = unsafe_points[0][1]
            if safe_point_max_succ_times_2 == 0:
                quadrant = None
            else:
                quadrant = (safe_point_2.y - curr_loc.y, safe_point_2.x - curr_loc.x)
            desire_distance_2 = get_distance(curr_loc, safe_point_2)
            desire_angle_2 = sympy.atan2(safe_point_2.y - curr_loc.y, safe_point_2.x - curr_loc.x)

        if middle_point_max_succ_times == -1:
            # try to go to sample around middle points
            desire_distance_2, desire_angle_2, middle_point_max_succ_times = self.go_for_middle_points_in_circle(
                curr_loc,
                radius,
                golf_map,
                quadrant,
                desire_angle_2,
                max_possible_score, target)
            if safe_point_max_succ_times_2 >= middle_point_max_succ_times:
                prev_sample_point_2 = safe_point_2
                self.logger.info(str(self.turn) + "still choose sample points")
                desire_distance_2 = get_distance(curr_loc, safe_point_2)
                desire_angle_2 = sympy.atan2(safe_point_2.y - curr_loc.y, safe_point_2.x - curr_loc.x)
            else:
                self.logger.info(str(self.turn) + "choose new sample points around the middle points")

        if middle_point_max_succ_times > safe_point_max_succ_times_2:
            if desire_distance_1 and desire_angle_1:
                desire_distance_2, desire_angle_2 = desire_distance_1, desire_angle_1

        now_max_succ_times_2 = max(safe_point_max_succ_times_2, middle_point_max_succ_times)
        if now_max_succ_times > now_max_succ_times_2:
            if prev_sample_point_1:
                self.prev_sample_point.append(prev_sample_point_1)
            self.logger.info(
                str(self.turn) + "choose max_possible_score < curr_expected_score" + str(now_max_succ_times))
            return desire_distance_1, desire_angle_1

        if prev_sample_point_2:
            self.prev_sample_point = prev_sample_point_2
        self.logger.info(
            str(self.turn) + "choose max_possible_score == curr_expected_score" + str(now_max_succ_times_2))
        return (desire_distance_2, desire_angle_2)

    def go_for_middle_points_in_circle(self, curr_loc, radius, golf_map, quadrant, desire_angle, max_possible_score,
                                       target):
        # 2. if we cannot use greedy, we try to find the points intersected with the golf map
        if quadrant:
            quadranty, quadrantx = quadrant
        circle = sympy.Circle(curr_loc, radius)
        intersect_points_origin = circle.intersection(golf_map)

        intersect_points_num = len(intersect_points_origin)
        temp_middle_points = []

        for i in range(intersect_points_num):
            for j in range(i + 1, intersect_points_num):
                middle_point = shapely.geometry.Point(
                    float(intersect_points_origin[i].x + intersect_points_origin[j].x) / 2,
                    float(intersect_points_origin[i].y + intersect_points_origin[j].y) / 2)
                # find points that in the golf map polygon
                if self.shapely_golf_map.contains(middle_point):
                    temp_middle_points.append(middle_point)

        if len(temp_middle_points) == 0:
            self.logger.error(str(self.turn) + "cannot find any middle point")
            return (radius, desire_angle, 0)

        # if there are many ways to go, delete the points that can go back
        middle_points = []
        if quadrant and desire_angle:
            for i, middle_point in enumerate(temp_middle_points):
                if quadrant is None:
                    expected_score = self.get_expected_score(self.point_dict, middle_point)
                    if expected_score > max_possible_score:
                        continue
                else:
                    curr_to_mid_quadranty, curr_to_mid_quadrantx = middle_point.y - curr_loc.y, middle_point.x - curr_loc.x
                    if curr_to_mid_quadranty * quadranty < 0 or curr_to_mid_quadrantx * quadrantx < 0:
                        continue

                middle_points.append(middle_point)
        else:
            middle_points = temp_middle_points

        # if there we delete every point in temp_middle_points,
        # which means we could go longer ways than expected, we need to add back those points
        if len(middle_points) == 0:
            self.logger.info(str(self.turn) + "cannot find any reasonable middle point!")
            return (radius, desire_angle, 0)

        middle_points_num = len(middle_points)
        curr2mid_angle = [0] * middle_points_num
        if desire_angle:
            for i, middle_point in enumerate(middle_points):
                curr_to_mid_angle = sympy.atan2(middle_point.y - curr_loc.y, middle_point.x - curr_loc.x)
                curr2mid_angle[i] = abs(curr_to_mid_angle - desire_angle)
        else:
            for i, middle_point in enumerate(middle_points):
                curr2mid_angle[i] = get_distance(middle_point, target)

        # rank the middle point from the closest angle to sample points to farther
        distance_sorted_indexes = sorted(range(middle_points_num), key=lambda x: curr2mid_angle[x])

        safe_point = None
        unsafe_points2score = dict()

        for i in distance_sorted_indexes:
            middle_point = middle_points[i]
            angle = sympy.atan2(middle_point.y - curr_loc.y, middle_point.x - curr_loc.x)
            succ_times = 0
            for _ in range(self.simulate_times):
                is_succ, _ = self.simulate_shapely_once(get_distance(middle_point, curr_loc), angle, curr_loc,
                                                        self.shapely_golf_map)
                succ_times += is_succ

            if succ_times / self.simulate_times >= 1 - self.risk:
                safe_point = middle_point
                break

            point = Point(middle_point.x, middle_point.y)
            unsafe_points2score[point] = succ_times

        if safe_point:
            self.logger.info(str(self.turn) + "select middle point to go")
            desire_distance = get_distance(safe_point, curr_loc)
            desire_angle = sympy.atan2(safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)
            return (desire_distance, desire_angle, self.simulate_times)

        # 3. if middle points are still not safe, choose the safest one and test its surrounding sample points
        unsafe_points = sorted(unsafe_points2score.items(), key=lambda x: -x[1])
        safe_middle_point = unsafe_points[0][0]

        desire_angle = sympy.atan2(safe_middle_point.y - curr_loc.y, safe_middle_point.x - curr_loc.x)

        delta_times = 100
        delta = radius / delta_times
        unsafe_distance2score = dict()
        desire_distance = None

        count = 0
        while True:
            delte_distance = radius - delta * count
            count += 1
            if count > delta_times / 2:
                break
            delta_middle_point = shapely.geometry.Point(curr_loc.x + delte_distance * sympy.cos(desire_angle),
                                                        curr_loc.y + delte_distance * sympy.sin(desire_angle))

            if quadrant or desire_angle:
                expected_score = self.get_expected_score(self.point_dict, delta_middle_point)
                if expected_score > max_possible_score:
                    continue

            if not self.shapely_golf_map.contains(delta_middle_point):
                continue

            succ_times = 0
            for _ in range(self.simulate_times):
                is_succ, _ = self.simulate_shapely_once(delte_distance, desire_angle, curr_loc,
                                                        self.shapely_golf_map)
                succ_times += is_succ

            if succ_times / self.simulate_times >= 1 - self.risk:
                desire_distance = delte_distance
                break

            unsafe_distance2score[delte_distance] = succ_times

        # deltax = 5
        # deltay = 5
        # dimension = 10
        # xrange = np.linspace(safe_middle_point.x - deltax, safe_middle_point.x + deltax, dimension)
        # yrange = np.linspace(safe_middle_point.y - deltay, safe_middle_point.y + deltay, dimension)
        #
        # safe_middle_angle = sympy.atan2(safe_middle_point.y - curr_loc.y, safe_middle_point.x - curr_loc.x)
        #
        # sample_points = dict()
        # for xi in range(len(xrange)):
        #     for yi in range(len(yrange)):
        #         sample_point = shapely.geometry.Point(xrange[xi], yrange[yi])
        #         difference_angle = abs(sympy.atan2(yrange[yi] - curr_loc.y, xrange[xi] - curr_loc.x) - safe_middle_angle)
        #         if self.shapely_golf_map.contains(sample_point):
        #             sample_point = Point(sample_point.x, sample_point.y)
        #             sample_points[sample_point] = difference_angle
        #
        # sample_points = dict(sorted(sample_points.items(), key=lambda x: x[1]))
        #
        # safe_point = None
        # unsafe_points2score = dict()
        # for point in sample_points.keys():
        #     succ_times = 0
        #     for _ in range(self.simulate_times):
        #         angle = sympy.atan2(point.y - curr_loc.y, point.x - curr_loc.x)
        #         is_succ, _ = self.simulate_shapely_once(get_distance(curr_loc, point), angle, curr_loc,
        #                                                 self.shapely_golf_map)
        #         succ_times += is_succ
        #
        #     if succ_times / self.simulate_times >= 1 - self.risk:
        #         safe_point = point
        #         break
        #
        #     unsafe_points2score[point] = succ_times

        if desire_distance is None:
            self.logger.info(str(self.turn) + "risky!!! select most safe point to middle point to go")
            if len(unsafe_distance2score) == 0:
                desire_distance = get_distance(safe_middle_point, curr_loc)
                max_succ_times = unsafe_points[0][1]
            else:
                unsafe_points = sorted(unsafe_distance2score.items(), key=lambda x: -x[1])
                desire_distance = unsafe_points[0][0]
                max_succ_times = unsafe_points[0][1]
        else:
            self.logger.info(str(self.turn) + "select safe distance to safest middle point to go")
            max_succ_times = self.simulate_times

        # desire_distance = get_distance(curr_loc, safe_point)
        # desire_angle = sympy.atan2(safe_point.y - curr_loc.y, safe_point.x - curr_loc.x)

        return (desire_distance, desire_angle, max_succ_times)

    def get_expected_score(self, points_score, curr_loc):
        expected_score = 100
        smallest_distance = sys.maxsize
        for point, score in points_score.items():
            current_dist = get_distance(curr_loc, point)
            if current_dist < smallest_distance:
                expected_score = score
                smallest_distance = current_dist
        return expected_score

    def get_expected_next_point(self, distance, angle, curr_loc):
        final_point = Point(curr_loc.x + distance * sympy.cos(angle),
                            curr_loc.y + distance * sympy.sin(angle))

        return final_point

    def go_for_sample_points_in_circle(self, curr_loc, radius, quadrant, desire_angle, max_possible_score):
        quadranty, quadrantx = quadrant
        quadranty = 1 if quadranty > 0 else -1
        quadrantx = 1 if quadrantx > 0 else -1
        sample_points = dict()
        dimension = 20
        if desire_angle % np.pi / 2 == 0:
            if desire_angle == 0 or desire_angle == np.pi:
                xmin, xmax = float(min(curr_loc.x, curr_loc.x + quadrantx * radius)), float(max(curr_loc.x,
                                                                                                curr_loc.x + quadrantx * radius))
                ymin, ymax = float(curr_loc.y - radius), float(curr_loc.y + radius)
                xrange = np.linspace(xmin, xmax, dimension)
                yrange = np.linspace(ymin, ymax, dimension * 2)

            else:
                xmin, xmax = float(curr_loc.x - radius), float(curr_loc.x + radius)
                ymin, ymax = float(min(curr_loc.y, curr_loc.y + quadranty * radius)), float(max(curr_loc.y,
                                                                                                curr_loc.y + quadranty * radius))
                xrange = np.linspace(xmin, xmax, dimension * 2)
                yrange = np.linspace(ymin, ymax, dimension)

            for xi in range(1, len(xrange)):
                for yi in range(1, len(yrange)):
                    sample_point = shapely.geometry.Point(xrange[xi], yrange[yi])
                    difference_angle = abs(sympy.atan2(yrange[yi] - curr_loc.y, xrange[xi] - curr_loc.x) - desire_angle)
                    if difference_angle >= np.pi / 4:
                        continue
                    if self.shapely_golf_map.contains(sample_point):
                        expected_score = self.get_expected_score(self.point_dict, sample_point)
                        if expected_score <= max_possible_score:
                            sample_point = Point(sample_point.x, sample_point.y)
                            sample_points[sample_point] = (expected_score, difference_angle)

        else:
            xmin, xmax = float(min(curr_loc.x, curr_loc.x + quadrantx * radius)), float(
                max(curr_loc.x, curr_loc.x + quadrantx * radius))
            ymin, ymax = float(min(curr_loc.y, curr_loc.y + quadranty * radius)), float(
                max(curr_loc.y, curr_loc.y + quadranty * radius))
            xrange = np.linspace(xmin, xmax, dimension)
            yrange = np.linspace(ymin, ymax, dimension)

            for xi in range(1, len(xrange)):
                for yi in range(1, len(yrange)):
                    sample_point = shapely.geometry.Point(xrange[xi], yrange[yi])
                    difference_angle = abs(sympy.atan2(yrange[yi] - curr_loc.y, xrange[xi] - curr_loc.x) - desire_angle)
                    if self.shapely_golf_map.contains(sample_point):
                        expected_score = self.get_expected_score(self.point_dict, sample_point)
                        if expected_score <= max_possible_score:
                            sample_point = Point(sample_point.x, sample_point.y)
                            sample_points[sample_point] = (expected_score, difference_angle)

        if len(sample_points) == 0:
            return None, None

        sample_points = dict(sorted(sample_points.items(), key=lambda x: (x[1][0], x[1][1])))

        safe_point = None
        unsafe_points2score = dict()
        for point in sample_points.keys():
            succ_times = 0
            for _ in range(self.simulate_times):
                angle = sympy.atan2(point.y - curr_loc.y, point.x - curr_loc.x)
                is_succ, _ = self.simulate_shapely_once(get_distance(curr_loc, point), angle, curr_loc,
                                                        self.shapely_golf_map)
                succ_times += is_succ

            if succ_times / self.simulate_times >= 1 - self.risk:
                safe_point = point
                break

            unsafe_points2score[point] = succ_times

        if safe_point is None:
            unsafe_points = sorted(unsafe_points2score.items(), key=lambda x: -x[1])
            safe_point = unsafe_points[0][0]
            safe_point_max_succ_times = unsafe_points[0][1]
        else:
            safe_point_max_succ_times = self.simulate_times

        return safe_point, safe_point_max_succ_times
