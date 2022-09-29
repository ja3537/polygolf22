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

# Sampling Size
SAMPLE_SIZE = 1200
GRANULARITY = 7
POINTS_ON_MAP_EDGE = 10
POINTS_ON_SAND_EDGE = 0
SANDTRAP_HEURISTIC = 1.1

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
    def __init__(self, point: Tuple[float, float], actual_cost=float('inf'), previous=None, next=None, skill=50, in_sandtrap = False):
        self.point = point

        self.previous = previous
        self.next = next

        #actual_cost will be the Expected number of shots to the Goal
        self._actual_cost = actual_cost

        max_target_dist = 200 + skill
        max_dist = standard_ppf(0.99) * (max_target_dist / skill) + max_target_dist
        max_dist *= 1.10

        self.in_sandtrap = in_sandtrap

    @property
    def actual_cost(self):
        return self._actual_cost

    def __lt__(self, other):
        return self.actual_cost < other.actual_cost

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
        self.sandtrap_points_set = set()

        self.sandtrap_mask = None

        self.mpl_sand_polys = None
        self.max_sand_ddist = scipy_stats.norm(max_dist / 2, (max_dist / self.skill)*2)

        #hash data for ev(a,b), key = (origin, dest), eg. ((1,1), (2,2)), value = EV((1,1,), (2,2))
        self.ev_hash = {}

        #hash data, key = (scored_point), value = next optimal point from scored_point
        self.best_cost_to_goal = {}

    def numpy_adjacent_and_dist (self, point: Tuple[float, float]):
        cloc_distances = cdist(self.np_map_points, np.array([np.array(point)]), 'euclidean')
        cloc_distances = cloc_distances.flatten()

        reachable_points = None
        if self.point_in_sandtrap_mpl(point):
            sand_trap_distance_mask = cloc_distances <= (200 + self.skill) / 2
            reachable_points = self.np_map_points[sand_trap_distance_mask]
        else:
            distance_mask = cloc_distances <= 200 + self.skill
            reachable_points = self.np_map_points[distance_mask]
        
        return reachable_points
    
    def point_in_sandtrap_mpl(self, current_point: Tuple[float, float]) -> bool:
        for sandtrap in self.mpl_sand_polys:
            if sandtrap.contains_point(current_point):
                return True
        
        return False

    def next_target(self, curr_loc: Tuple[float, float], distance: float) -> Union[None, Tuple[float, float]]:

        lowest_ev_value = float("inf")
        lowest_ev_point = None
        lowest_roll_dist_to_goal = float("inf")
        
        reachable_points = self.numpy_adjacent_and_dist(curr_loc)

        for i in range(len(reachable_points)):

            candidate_point = tuple(reachable_points[i])

            marginal_ev_of_hits_to_candidate_point = self.get_ev(curr_loc, candidate_point, self.skill, self.point_in_sandtrap_mpl(curr_loc))
            self.ev_hash[(curr_loc,candidate_point)] = marginal_ev_of_hits_to_candidate_point

            ev_value = marginal_ev_of_hits_to_candidate_point + self.best_cost_to_goal[candidate_point]

            if ev_value < lowest_ev_value:
                lowest_ev_value = ev_value
                lowest_ev_point = candidate_point

                lowest_roll_dist_to_goal = self.roll_point_dist_to_goal(curr_loc, candidate_point)

            #if its a tie, take take the lowest distance to the goal
            elif ev_value == lowest_ev_value:

                roll_dist_to_goal = self.roll_point_dist_to_goal(curr_loc, candidate_point)

                if roll_dist_to_goal < lowest_roll_dist_to_goal:
                    lowest_roll_dist_to_goal = roll_dist_to_goal
                    lowest_ev_point = candidate_point
        
        return lowest_ev_point

    def roll_point_dist_to_goal(self, curr_loc, target):
        #calculate point of roll
        x0, y0 = curr_loc
        x1, y1 = target
        xg, yg = self.goal

        #half the roll distance
        xr, yr = x1 + (x1-x0)*0.1, y1 + (y1-y0)*0.1

        return math.sqrt((xg-xr)**2+(yg-yr)**2)
        

    def _initialize_map_points(self, goal: Tuple[float, float], golf_map: Polygon, sand_traps: list[sympy.Polygon]):
        # Storing the points as numpy array
        np_map_points = [goal]
        np_sand_trap_points = []
        sandtrap_mask = [False]
        self.mpl_poly = sympy_poly_to_mpl(golf_map)
        self.shapely_poly = sympy_poly_to_shapely(golf_map)

        self.mpl_sand_polys = [sympy_poly_to_mpl(sand_trap) for sand_trap in sand_traps]

        pp = self.poly_to_points(poly = golf_map, mpl_poly = self.mpl_poly)
        for point in pp:

            x, y = point
            
            np_map_points.append(np.array([x, y]))

            in_sandtrap = False
            for sandtrap in self.mpl_sand_polys:
                if sandtrap.contains_point(point):
                    x, y = point
                    np_sand_trap_points.append(np.array([x, y]))
                    self.sandtrap_points_set.add(point)
                    in_sandtrap = True
            sandtrap_mask.append(in_sandtrap)

        #add points along edges of map  
        map_edges = self.polygon_edge_sampler(golf_map, POINTS_ON_MAP_EDGE)
        np_map_points += map_edges
        sandtrap_mask += [False]*len(map_edges)

        #add points along edges of sandtraps

        if POINTS_ON_SAND_EDGE != 0:
            for s in sand_traps:
                temp = self.polygon_edge_sampler(s, POINTS_ON_SAND_EDGE)

                #add the sand trap edges to both the map_points and sand_trap_points
                np_map_points += temp
                np_sand_trap_points += temp
        self.np_map_points = np.array(np_map_points)
        self.np_sand_trap_points = np.array(np_sand_trap_points)
        self.np_goal_dist = cdist(self.np_map_points, np.array([np.array(self.goal)]), 'euclidean')
        self.np_goal_dist = self.np_goal_dist.flatten()
        self.sandtrap_mask = np.array(sandtrap_mask)

        self.a_star(goal)

    def a_star(self, goal):

        point_goal = goal
        heap = [ScoredPoint(point=point_goal, actual_cost=0.0, previous=None, skill=self.skill)]

        #optimization
        visited = set()
        in_queue = set()
        best_cost_to_goal = {}

        while len(heap) > 0:
            next_sp = heapq.heappop(heap)
            next_p = next_sp.point


            #check if the point has been visited
            if next_p in visited:
                continue

            #if it is greater than 10, and has not been seen before, label the point in best_cost_to_goal
            if next_sp.actual_cost > 10:
                best_cost_to_goal[next_p] = next_sp.actual_cost
                continue
                
            #tag the point is visited
            visited.add(next_p)
            best_cost_to_goal[next_p] = next_sp.actual_cost

            #, distance
            reachable_points = self.astar_max_reachable_points(next_p)

            #reachable points
            for i in range(len(reachable_points)):
                
                curr_point = tuple(reachable_points[i])

                #if we have already seen a point, skip it
                if curr_point in visited or curr_point in in_queue:
                    continue

                #calculate the expected value of hitting from current_loc to the curr_point
                marginal_ev_of_hits_to_candidate_point = self.get_ev(curr_point, next_p, self.skill, next_p in self.sandtrap_points_set)
                self.ev_hash[(curr_point, next_p)] = marginal_ev_of_hits_to_candidate_point

                #if we have seen the best cost of this point, and the current cost is worse, skip it
                if curr_point in best_cost_to_goal and best_cost_to_goal[curr_point] < next_sp.actual_cost + marginal_ev_of_hits_to_candidate_point:
                    continue

                #(self, point: Tuple[float, float], actual_cost=float('inf'), previous=None, skill=50, in_sandtrap = False):
                new_point = ScoredPoint(point = curr_point,
                                        actual_cost = next_sp.actual_cost + marginal_ev_of_hits_to_candidate_point,
                                        next = next_p,
                                        skill=self.skill)

                best_cost_to_goal[curr_point] = next_sp.actual_cost + marginal_ev_of_hits_to_candidate_point

                heapq.heappush(heap, new_point)
        
        best_cost_to_goal[self.goal] = 1
        self.best_cost_to_goal = best_cost_to_goal

    def astar_max_reachable_points(self, point: Tuple[float, float]):

        cloc_distances = cdist(self.np_map_points, np.array([np.array(point)]), 'euclidean')
        cloc_distances = cloc_distances.flatten()

        cloc_distances[self.sandtrap_mask] *= 2

        distance_mask = cloc_distances <= 200 + self.skill

        reachable_points = self.np_map_points[distance_mask]
            
        return reachable_points
            
    
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
        
        return result

    def polygon_edge_sampler(self, poly: Polygon, points_per_edge: int) -> list[np.array([float, float])]:
        
        if points_per_edge == 0:
            return []
        
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
        cl = float(curr_loc.x), float(curr_loc.y)
        current_point = np.array(tuple(curr_loc)).astype(float)

        dist_to_goal = np.linalg.norm(np.array(self.goal) - current_point)
        target_point = self.next_target(cl, dist_to_goal)

        # fixup target
        original_dist = np.linalg.norm(np.array(target_point) - current_point)
        in_sand = self.point_in_sandtrap_mpl(current_point)

        if tuple(target_point) == self.goal:
            v = np.array(target_point) - current_point
            # Unit vector pointing from current to target
            u = v / original_dist

            if original_dist <= 20.0 and in_sand == False:
                if self.skill >= 60:
                    target_point = current_point + u * (original_dist * 1.25)
                elif self.skill >= 40:
                    target_point = current_point + u * (original_dist * 1.1)
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
        granularity = GRANULARITY

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

        land_total_prob = np.sum(joint_dist_pdf, where=joint_cord_is_land) / joint_total_prob
        water_prob = 1 - land_total_prob
        sand_prob = np.sum(joint_dist_pdf, where=joint_cord_is_sand) / joint_total_prob
        green_prob = land_total_prob - sand_prob

        if water_prob > 0.90:
            return 11
        
        expected_tries_to_hit_land = land_total_prob**(-1)          # if probability of hitting land is 0.25, we expect 0.25**(-1) = 4 tries to hit land
        
        expected_value = (water_prob * expected_tries_to_hit_land) + (sand_prob * SANDTRAP_HEURISTIC) + (green_prob * 1)

        return expected_value
