import logging
import math
import random
from collections import deque
from typing import Tuple, List, Dict, Optional

import os
import pickle
import numpy as np
import sympy
from numba import njit
from scipy.spatial import KDTree

import constants

SAMPLE_LIMIT = 3000  # approx. count of sampled points
MINIMUM_SAMPLE_DISTANCE = 20  # minimum distance between sampled points

RANDOM_COUNT = 60  # repeat times of sampling normal distributions
PRUNING_FACTOR = 0.2

EPS = 1e-12


class PointF:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return f"P({self.x}, {self.y})"

    def __repr__(self):
        return f"PointF({self.x}, {self.y})"

    def __eq__(self, other):
        return dist(self, other) < EPS

    def __hash__(self):
        return hash(self.x) ^ hash(self.y)

    def __sub__(self, other):
        assert type(other) == PointF
        return PointF(self.x - other.x, self.y - other.y)

    @property
    def to_numpy(self):
        return np.array([self.x, self.y])


def to_numeric_point(p: sympy.Point2D) -> PointF:
    return PointF(p.x, p.y)


def dist2(p1: PointF, p2: PointF) -> float:
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2


def dist(p1: PointF, p2: PointF = PointF(0, 0)) -> float:
    return math.sqrt(dist2(p1, p2))


def dist_to_seg(p: PointF, s: PointF, t: PointF) -> float:
    def sgn(x: float) -> int:
        if math.fabs(x) < EPS:
            return 0
        return 1 if x > 0 else -1

    def dot(a: PointF, b: PointF) -> float:
        return a.x * b.x + a.y * b.y

    def det(a: PointF, b: PointF) -> float:
        return a.x * b.y - a.y * b.x

    def cross(s: PointF, t: PointF, o: PointF = PointF(0, 0)) -> float:
        return det(s - o, t - o)

    def dist_to_line(p: PointF, s: PointF, t: PointF) -> float:
        return math.fabs(cross(s, t, p)) / dist(s - t)

    if s == t:
        return dist(p, s)
    vs = p - s
    vt = p - t
    if sgn(dot(t - s, vs)) < 0:
        return dist(vs)
    elif sgn(dot(t - s, vt)) > 0:
        return dist(vt)
    return dist_to_line(p, s, t)


@njit
def point_inside_polygon(poly: np.array, px: float, py: float) -> bool:
    # http://paulbourke.net/geometry/polygonmesh/#insidepoly
    # https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if min(p1y, p2y) < py <= max(p1y, p2y) and px <= max(p1x, p2x) and p1x != p2y:
            xints = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or px <= xints:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


@njit
def sgn_cross(o: np.array, a: np.array, b: np.array) -> int:
    t = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    if np.fabs(t) < EPS:
        return 0
    return 1 if t > 0 else -1


@njit
def segment_polygon_intersection(poly: np.array, b1: np.array, b2: np.array) -> bool:
    n = len(poly)
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        d1 = sgn_cross(a1, a2, b1)
        d2 = sgn_cross(a1, a2, b2)
        d3 = sgn_cross(b1, b2, a1)
        d4 = sgn_cross(b1, b2, a2)
        if d1 ^ d2 == -2 and d3 ^ d4 == -2:
            return True
    return False


def sample_points_inside_polygon(poly: sympy.Polygon, poly_f: np.array) -> Tuple[float, List[PointF]]:
    def sample_by_dist(d: float) -> List[PointF]:
        l = list()
        xmin, ymin, xmax, ymax = poly.bounds
        for x in np.arange(float(xmin), float(xmax), d):
            for y in np.arange(float(ymin), float(ymax), d):
                p = PointF(x, y)
                if point_inside_polygon(poly_f, p.x, p.y):
                    l.append(p)
        return l

    dist = MINIMUM_SAMPLE_DISTANCE
    while True:
        s = sample_by_dist(dist)
        if len(s) > SAMPLE_LIMIT:
            return dist, s
        dist /= math.sqrt(2)  # increase the number of sampled points by 2


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
        self.skill = skill
        self.rng = rng
        self.logger = logger

        self.max_dist = constants.max_dist + self.skill

        self.need_initialization = False

        # for numeric computation
        golf_map_f = list()
        for v in golf_map.vertices:
            golf_map_f.append(to_numeric_point(v))
        self.target_f = target_f = to_numeric_point(target)
        self.golf_map_f = np.asarray([(p.x, p.y) for p in golf_map_f], dtype=np.float64)

        precomp_path = os.path.join(precomp_dir, "{}_skill-{}.pkl".format(map_path, skill))
        if os.path.isfile(precomp_path):
            with open(precomp_path, "rb") as f:
                self.sample_dist, self.sampled_points, self.scores, self.kdt = pickle.load(f)
        else:
            self.sample_dist, self.sampled_points = sample_points_inside_polygon(golf_map, self.golf_map_f)

            # calculate scores
            self.scores: Dict[PointF, float] = dict()
            self.calc_scores(target_f, self.max_dist)

            # build KD-Tree
            self.kdt = KDTree([(p.x, p.y) for p in self.sampled_points])

            with open(precomp_path, 'wb') as f:
                pickle.dump([self.sample_dist, self.sampled_points, self.scores, self.kdt], f)

        self.logger.debug(f"# of sampled points: {len(self.sampled_points)}")
        self.logger.debug(f"max score: {max(self.scores.values())}")

    def calc_scores(self, target: PointF, max_d: float):
        # naive BFS
        queue = deque([target])
        self.scores[target] = 0

        while queue:
            cur = queue.popleft()
            cur_score = self.scores[cur]
            for p in self.sampled_points:
                if p in self.scores or dist(p, cur) >= max_d:
                    continue
                self.scores[p] = cur_score + 1 + 1e-10 * dist(p, cur)
                queue.append(p)

    def score(self, p: PointF) -> float:
        # return the score of nearest point
        d, idx = self.kdt.query([p.x, p.y], k=1)
        if d > self.sample_dist:
            return float('inf')
        return self.scores.get(self.sampled_points[idx])

    @staticmethod
    def pos(current_position: PointF, distance: float, angle: float) -> PointF:
        return PointF(current_position.x + distance * math.cos(angle),
                      current_position.y + distance * math.sin(angle))

    def evaluate_putter(self, current_position: PointF, distance: float, angle: float) -> Optional[float]:
        end = self.pos(current_position, distance, angle)

        # boundary check
        if not point_inside_polygon(self.golf_map_f, end.x, end.y):
            return None
        if segment_polygon_intersection(self.golf_map_f, current_position.to_numpy, end.to_numpy):
            return None

        # goal
        if dist_to_seg(self.target_f, current_position, end) < constants.target_radius:
            return 0

        return self.score(end)

    def evaluate(self, current_position: PointF, distance: float, angle: float) -> Optional[float]:
        return self.evaluate_putter(self.pos(current_position, distance, angle),
                                    distance * constants.extra_roll,
                                    angle)

    def simulate(self, candidates: List[Tuple[float, float]], current_position: PointF) -> Tuple[Tuple[float, float], float]:
        min_score = float('inf')

        def get_score(scores: List[float]) -> float:
            n = len(scores)
            valid_scores = list(filter(lambda x: x is not None, scores))
            if not valid_scores:
                return float('inf')
            avg = sum(valid_scores) / len(valid_scores)
            miss_prob = (n - len(valid_scores)) / n
            return avg + miss_prob / (1 - miss_prob)

        def simulate_one(candidate: Tuple[float, float]) -> Optional[float]:
            distance, angle = candidate
            scores = list()
            for counter in range(RANDOM_COUNT):
                # pruning
                if counter == int(RANDOM_COUNT * PRUNING_FACTOR):
                    if get_score(scores) > min_score * 1.5:
                        return None

                real_distance = np.random.normal(distance, distance / self.skill)
                real_angle = angle + np.random.normal(0, 1 / (2 * self.skill))
                if distance < constants.min_putter_dist:  # putter
                    scores.append(self.evaluate_putter(current_position, real_distance, real_angle))
                else:
                    scores.append(self.evaluate(current_position, real_distance, real_angle))
            return get_score(scores)

        res = None
        for candidate in candidates:
            v = simulate_one(candidate)
            if not v or v > min_score:
                continue
            min_score = v
            res = candidate
        return res, min_score

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
        curr_loc = to_numeric_point(curr_loc)
        target_angle = math.atan2(self.target_f.y - curr_loc.y, self.target_f.x - curr_loc.x)

        candidates = [
            (distance, angle)
            for distance in list(range(1, 20)) + list(range(20, self.max_dist, 3)) + [self.max_dist]
            for angle in [2 * math.pi * (i / 72) for i in range(72)] + [target_angle]
        ]
        random.shuffle(candidates)

        choice, score = self.simulate(candidates, curr_loc)

        # naive method for special case
        distance, _ = choice
        if distance < constants.min_putter_dist:
            return min(dist(self.target_f, curr_loc) / (1 - 1 / self.skill * 3), constants.min_putter_dist), target_angle

        self.logger.debug(f"last: {to_numeric_point(prev_loc) if prev_loc else prev_loc}, target: {target}")
        self.logger.debug(f"choice: {choice}, score: {score}")

        return choice
