'''import numpy as np
import sympy
import logging
from typing import Tuple, List

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

'''
import os
import pickle
import numpy as np
import functools
import sympy
import logging
import heapq
from scipy import stats as scipy_stats
import math

from typing import Tuple, Iterator, List, Union
from sympy.geometry import Polygon, Point2D
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from scipy.spatial.distance import cdist


# Cached distribution
DIST = scipy_stats.norm(0, 1)

#Sampling Size
#SAMPLE_SIZE = 0
SAMPLE_SIZE = 1000

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


def splash_zone(distance: float, angle: float, conf: float, skill: int, current_point: Tuple[float, float], current_in_sandtrap: bool = False, target_in_sandtrap: bool = False) -> np.array:
    conf_points = np.linspace(1 - conf, conf, 5)
    distances = np.vectorize(standard_ppf)(conf_points) * (distance / skill) + distance

    angle_factor = 2 if current_in_sandtrap else 1

    angles = np.vectorize(standard_ppf)(conf_points) * (1*angle_factor/(2*skill)) + angle

    scale = 1.1

    #when in in_sandtrap, do not account for rolling by extending splash zone
    if distance <= 20 or target_in_sandtrap:
        scale = 1.0
    max_distance = distances[-1]*scale
    top_arc = spread_points(current_point, angles, max_distance, False)

    if distance > 20:
        min_distance = distances[0]
        bottom_arc = spread_points(current_point, angles, min_distance, True)
        return np.concatenate((top_arc, bottom_arc, np.array([top_arc[0]])))

    current_point = np.array([current_point])
    return np.concatenate((current_point, top_arc, current_point))



def sympy_poly_to_mpl(sympy_poly: Polygon) -> Path:
    """Helper function to convert sympy Polygon to matplotlib Path object"""
    v = list(sympy_poly.vertices)
    v.append(v[0])
    return Path(v, closed=True)


def sympy_poly_to_shapely(sympy_poly: Polygon) -> ShapelyPolygon:
    """Helper function to convert sympy Polygon to shapely Polygon object"""
    v = sympy_poly.vertices
    v.append(v[0])
    return ShapelyPolygon(v)


class ScoredPoint:
    """Scored point class for use in A* search algorithm"""
    def __init__(self, point: Tuple[float, float], goal: Tuple[float, float], actual_cost=float('inf'), previous=None, goal_dist=None, skill=50, in_sandtrap = False):
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

        self.in_sandtrap = in_sandtrap

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

        # Cached data
        max_dist = 200 + self.skill
        self.max_ddist = scipy_stats.norm(max_dist, max_dist / self.skill)

        self.np_sand_trap_points = None
        self.mpl_sand_polys = None
        self.max_sand_ddist = scipy_stats.norm(max_dist / 2, (max_dist / self.skill)*2)

        #hash data for ev(a,b), key = (origin, dest), eg. ((1,1), (2,2)), value = EV((1,1,), (2,2))
        self.ev_hash = {}


        #hash data, key = (scored_point), value = next optimal point from scored_point
        self.optimal_next = {}

        # Conf level
        self.conf = 0.95
        if self.skill < 40:
            self.conf = 0.75

    @functools.lru_cache()
    def _max_ddist_ppf(self, conf: float):
        return self.max_ddist.ppf(0.05)

    @functools.lru_cache()
    def _max_sand_ddist_ppf(self, conf: float):
        return self.max_sand_ddist.ppf(0.05)

    def reachable_point(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float) -> bool:
        """Determine whether the point is reachable with confidence [conf] based on our player's skill"""

        if type(current_point) == Point2D:
            current_point = tuple(current_point)
        if type(target_point) == Point2D:
            target_point = tuple(target_point)

        current_point = np.array(current_point).astype(float)
        target_point = np.array(target_point).astype(float)

        if self.point_in_sandtrap_mpl(current_point):
            return np.linalg.norm(current_point - target_point) <= (200 + self.skill) / 2
            # return np.linalg.norm(current_point - target_point) <= self._max_sand_ddist_ppf(conf)
        else:
            return np.linalg.norm(current_point - target_point) <= 200 + self.skill
            # return np.linalg.norm(current_point - target_point) <= self._max_ddist_ppf(conf)

    def splash_zone_within_polygon(self, current_point: Tuple[float, float], target_point: Tuple[float, float], conf: float) -> bool:
        if type(current_point) == Point2D:
            current_point = tuple(Point2D)

        if type(target_point) == Point2D:
            target_point = tuple(Point2D)

        #CHANGES: checks if point shot from is in sandtrap
        current_in_sandtrap = False
        if  current_point in np.array(self.np_sand_trap_points):
            current_in_sandtrap = True

        #CHANGES: checks if landing point is in sandtrap
        target_in_sandtrap = False
        if  target_point in np.array(self.np_sand_trap_points):
            target_in_sandtrap = True

        distance = np.linalg.norm(np.array(current_point).astype(float) - np.array(target_point).astype(float))
        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(float(ty) - float(cy), float(tx) - float(cx))

        #CHANGES: add in_sandtrap
        splash_zone_poly_points = splash_zone(float(distance), float(angle), float(conf), self.skill, current_point, current_in_sandtrap, target_in_sandtrap)
        return self.shapely_poly.contains(ShapelyPolygon(splash_zone_poly_points))

    def numpy_adjacent_and_dist (self, point: Tuple[float, float]):
        cloc_distances = cdist(self.np_map_points, np.array([np.array(point)]), 'euclidean')
        cloc_distances = cloc_distances.flatten()

        distance_mask = cloc_distances <= 200 + self.skill
        sand_trap_distance_mask = cloc_distances <= (200 + self.skill) / 2

        reachable_points = None
        if self.point_in_sandtrap_mpl(point):
            reachable_points = self.np_map_points[sand_trap_distance_mask]
        else:
            reachable_points = self.np_map_points[distance_mask]
        
        goal_distances = self.np_goal_dist[distance_mask]
        return reachable_points, goal_distances
    
    def point_in_sandtrap_mpl(self, current_point: Tuple[float, float]) -> bool:
        for sandtrap in self.mpl_sand_polys:
            if sandtrap.contains_point(current_point):
                return True
        
        return False

    def next_target(self, curr_loc: Tuple[float, float], goal: Point2D) -> Union[None, Tuple[float, float]]:
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
            
            """
            if next_sp.actual_cost > 0 and not self.splash_zone_within_polygon(next_sp.previous.point, next_p, conf):

                #if we have already seen a quicker way to get to next_p
                if next_p in best_cost:
                    del best_cost[next_p]
                continue
            """
            visited.add(next_p)

            if np.linalg.norm(np.array(self.goal) - np.array(next_p)) <= 5.4 / 100.0:
                # All we care about is the next point
                # TODO: We need to check if the path length is <= 10, because if it isn't we probably need to
                #  reduce the conf and try again for a shorter path.
                while next_sp.previous.point != start_point:
                    next_sp = next_sp.previous
                return next_sp.point
            
            # Add adjacent points to heap
            reachable_points, goal_dists = self.numpy_adjacent_and_dist(next_p)
            for i in range(len(reachable_points)):
                candidate_point = tuple(reachable_points[i])
                goal_dist = goal_dists[i]

                marginal_ev_of_hits_to_candidate_point = None

                if (next_p,candidate_point) in self.ev_hash:
                    marginal_ev_of_hits_to_candidate_point = self.ev_hash[(next_p,candidate_point)]
                else:
                    marginal_ev_of_hits_to_candidate_point = self.get_ev(next_sp.point, candidate_point, self.skill, self.point_in_sandtrap_mpl(next_sp.point))
                    self.ev_hash[(next_p,candidate_point)] = marginal_ev_of_hits_to_candidate_point

                new_point = ScoredPoint(candidate_point, point_goal, next_sp.actual_cost + marginal_ev_of_hits_to_candidate_point, next_sp,
                                        goal_dist=goal_dist, skill=self.skill)

                if candidate_point not in best_cost or best_cost[candidate_point] > new_point.actual_cost:
                    points_checked += 1
                    # if not self.splash_zone_within_polygon(new_point.previous.point, new_point.point, conf):
                    #     continue
                    best_cost[new_point.point] = new_point.actual_cost
                    heapq.heappush(heap, new_point)

        # No path available
        print(reachable_points)
        return None

    def _initialize_map_points(self, goal: Tuple[float, float], golf_map: Polygon, sand_traps: list[sympy.Polygon]):
        # Storing the points as numpy array
        np_map_points = [goal]
        np_sand_trap_points = []
        self.mpl_poly = sympy_poly_to_mpl(golf_map)
        self.shapely_poly = sympy_poly_to_shapely(golf_map)

        self.mpl_sand_polys = [sympy_poly_to_mpl(sand_trap) for sand_trap in sand_traps]

        pp = self.poly_to_points(poly = golf_map, mpl_poly = self.mpl_poly)
        for point in pp:
            # Use matplotlib here because it's faster than shapely for this calculation...
            """if self.mpl_poly.contains_point(point):
                # map_points.append(point)
                x, y = point
                np_map_points.append(np.array([x, y]))"""

            x, y = point
            np_map_points.append(np.array([x, y]))

            for sandtrap in self.mpl_sand_polys:
                if sandtrap.contains_point(point):
                    x, y = point
                    np_sand_trap_points.append(np.array([x, y]))

        #add points along edges of map  
        np_map_points += self.polygon_edge_sampler(golf_map, 5)

        #add points along edges of sandtraps
        for s in sand_traps:
            temp = self.polygon_edge_sampler(s, 5)

            #add the sand trap edges to both the map_points and sand_trap_points
            np_map_points += temp
            np_sand_trap_points += temp

        self.np_map_points = np.array(np_map_points)
        self.np_sand_trap_points = np.array(np_sand_trap_points)
        self.np_goal_dist = cdist(self.np_map_points, np.array([np.array(self.goal)]), 'euclidean')
        self.np_goal_dist = self.np_goal_dist.flatten()
    
    def poly_to_points(self, poly: Polygon, mpl_poly: Path) -> list[Tuple[float, float]]:

        result = []

        #finds the rectangular boundry of the map
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        for point in poly.vertices:
            x = float(point.x)
            y = float(point.y)
            x_min = min(x, x_min)
            x_max = max(x, x_max)
            y_min = min(y, y_min)
            y_max = max(y, y_max)
        #x_step = X_STEP
        #y_step = Y_STEP

        point_count = 0

        while point_count < SAMPLE_SIZE:
            x_rand = self.rng.uniform(x_min, x_max)
            y_rand = self.rng.uniform(y_min, y_max)

            p = (x_rand, y_rand)

            if mpl_poly.contains_point(p):
                point_count += 1
                result.append(p)

        """x_current = x_min
        y_current = y_min
        while x_current < x_max:
            while y_current < y_max:
                yield float(x_current), float(y_current)
                y_current += y_step
            y_current = y_min
            x_current += x_step"""
        
        return result

    def polygon_edge_sampler(self, poly: Polygon, points_per_edge: int) -> list[np.array([float, float])]:
        result = []

        v = poly.vertices
        for p in range(len(v)):
            start = v[p-1]
            end = v[p]

            x_val = np.linspace(start = float(start.x), stop = float(end.x), num = points_per_edge, dtype=float)
            y_val = np.linspace(start = float(start.y), stop = float(end.y), num = points_per_edge, dtype=float)
            result += list(np.array((x_val,y_val)).T)

        return result

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
        target_point = self.next_target(cl, target)

        # fixup target
        current_point = np.array(tuple(curr_loc)).astype(float)
        original_dist = np.linalg.norm(np.array(target_point) - current_point)
        if tuple(target_point) == self.goal:
            v = np.array(target_point) - current_point
            # Unit vector pointing from current to target
            u = v / original_dist
            in_sand = self.point_in_sandtrap_mpl(current_point)
            if original_dist >= 20.0:
                roll_distance = original_dist * 0.1
                max_offset = roll_distance
                offset = 0
                prev_target = target_point
                while offset < max_offset * .5 and self.splash_zone_within_polygon(tuple(current_point), target_point, confidence):
                    offset += 1
                    dist = original_dist - offset
                    prev_target = target_point
                    target_point = current_point + u * dist
                target_point = prev_target
            elif original_dist < 20 and in_sand == False:
                if self.skill >= 60:
                    target_point = current_point + u * (original_dist * 1.5)
                elif self.skill >= 40:
                    target_point = current_point + u * (original_dist * 1.2)
                elif self.skill >= 20:
                    target_point = current_point + u * (original_dist * 1)
                else:
                    if original_dist > 10:
                        target_point = current_point + u * (original_dist * .9)
                    elif original_dist > 1: 
                        target_point = current_point + u * (original_dist * .95)
                    else:
                        target_point = current_point + u * (original_dist * 1)
        cx, cy = current_point
        tx, ty = target_point
        angle = np.arctan2(ty - cy, tx - cx)

        rv = curr_loc.distance(Point2D(target_point, evaluate=False)), angle
        if original_dist < 20 and in_sand == False and rv[0] > 20:
            rv = 19.99, angle
        self.prev_rv = rv
        return rv
        
    def get_ev(self, origin: Tuple[float, float], dest: Tuple[float, float], skill, origin_in_sand: bool):
        # assumes dest is reachable with the current distance rating
        granularity = 5

        o_x, o_y = origin
        d_x, d_y = dest

        # distance
        distance = math.sqrt((o_x - d_x)**2 + (o_y - d_y)**2)
        d_dist_stdev = distance / skill                                                     # standard dev
        if origin_in_sand:
            d_dist_stdev *= 2
        
        if d_dist_stdev == 0.0:
            d_dist_samples = np.array([distance])
            d_dist_pdf = np.array([1])
        else:
            d_dist = scipy_stats.norm(distance, d_dist_stdev)                                   # distance distribution
            d_dist_samples = np.linspace(d_dist.ppf(0.01), d_dist.ppf(0.99), granularity)       # evenly spaced points in distribution
            d_dist_pdf = d_dist.pdf(d_dist_samples) / np.sum(d_dist.pdf(d_dist_samples))        # probability corresponding to each point (normalized)

        # angle
        angle = np.arctan2(d_y - o_y, d_x - o_x)
        a_dist_stdev = 1 / (2 * skill)                                                      # standard dev
        if origin_in_sand:
            a_dist_stdev *= 2
        a_dist = scipy_stats.norm(angle, a_dist_stdev)                                      # angle distribution
        a_dist_samples = np.linspace(a_dist.ppf(0.01), a_dist.ppf(0.99), granularity)       # evenly spaced points in distribution
        a_dist_pdf = a_dist.pdf(a_dist_samples) / np.sum(a_dist.pdf(a_dist_samples))        # probability corresponding to each point (normalized)
        
        # combine distance and angle into joint distribution
        cos_a_dist_samples = np.cos(a_dist_samples)
        sin_a_dist_samples = np.sin(a_dist_samples)
        joint_x = (np.outer(d_dist_samples, cos_a_dist_samples) + origin[0]).flatten()
        joint_y = (np.outer(d_dist_samples, sin_a_dist_samples) + origin[1]).flatten()
        joint_cords = np.array((joint_x, joint_y)).T

        # check rolling
        middle_idx = granularity // 2
        roll_x_dist = (distance / 10) * cos_a_dist_samples[middle_idx]
        roll_y_dist = (distance / 10) * sin_a_dist_samples[middle_idx]

        roll_start = dest
        roll_end = [roll_start[0] + roll_x_dist, roll_start[1] + roll_y_dist]
        roll_vertecies = [roll_start, roll_end, roll_start]

        roll_path = Path(roll_vertecies, closed=True)

        # if landing point not in sand, and the rolling path is not COMPLETELY contained by any sort of land, return "impossible"
        if(not self.point_in_sandtrap_mpl(dest) and not self.mpl_poly.contains_path(roll_path)):
            return 11
        
        joint_dist_pdf = np.outer(d_dist_pdf, a_dist_pdf).flatten()
        joint_total_prob = np.sum(joint_dist_pdf)

        joint_cord_is_sand = None
        for sandtrap in self.mpl_sand_polys:
            if joint_cord_is_sand is None:
                joint_cord_is_sand = sandtrap.contains_points(joint_cords)
            
            else:
                joint_cord_is_sand = np.logical_or(joint_cord_is_sand, sandtrap.contains_points(joint_cords))
        
        joint_cord_is_land = self.mpl_poly.contains_points(joint_cords)
        #joint_cord_is_water = np.logical_not(joint_cord_is_land)

        land_total_prob = np.sum(joint_dist_pdf, where=joint_cord_is_land) / joint_total_prob
        water_prob = 1 - land_total_prob
        sand_prob = np.sum(joint_dist_pdf, where=joint_cord_is_sand) / joint_total_prob
        green_prob = land_total_prob - sand_prob

        '''print("water_prob: " + str(water_prob))
        print("sand_prob: " + str(sand_prob))
        print("green_prob: " + str(green_prob))
        print("Total Prob: " + str(water_prob + sand_prob + green_prob))'''

        if water_prob > 0.9999:
            return 11
        
        expected_tries_to_hit_land = land_total_prob**(-1)          # if probability of hitting land is 0.25, we expect 0.25**(-1) = 4 tries to hit land
        
        expected_value = (water_prob * expected_tries_to_hit_land) + (sand_prob * 1) + (green_prob * 1)

        return expected_value


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
