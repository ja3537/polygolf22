import logging
import multiprocessing
from itertools import product
from time import perf_counter
from typing import List, Tuple

import mdptoolbox
import numpy as np
import sympy
from scipy.stats import norm
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

# =====================
# Problem Configuration
# =====================
#
x_quant = 25
y_quant = 25
# Distance rating: 200 + s
dist_quant = 20
angle_quant = 36

# =================
# Utility functions
# =================
#
def sympy_poly_to_shapely(sympy_poly: sympy.Polygon) -> ShapelyPolygon:
    """Helper function to convert sympy Polygon to shapely Polygon object"""
    v = list(sympy_poly.vertices)
    v.append(v[0])
    return ShapelyPolygon(v)


def to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


# ======
# Player
# ======
#
class Player:
    def to_bin_index(self, x, y):
        # xi = int((x - min_x) / (x_tick))
        # yi = int((y - min_y) / (y_tick))
        xi = next(xi for xi, x_bin in enumerate(self.x_bins) if x_bin > x) - 1
        yi = next(yi for yi, y_bin in enumerate(self.y_bins) if y_bin > y) - 1
        return xi, yi

    # ===========================
    # Shot transition calculation
    # ===========================
    #
    # distance/angle space -> x/y space?
    def sample_shots(
        self,
        start_x,
        start_y,
        skill,
        distance,
        angle,
        in_sand,
        num_samples=100,
    ):
        distance_dev = distance / skill
        angle_dev = 1 / (2 * skill)

        if in_sand:
            distance_dev *= 2
            angle_dev *= 2

        dist_rv = norm(loc=distance, scale=distance_dev)
        angle_rv = norm(loc=angle, scale=angle_dev)
        ds = dist_rv.rvs(size=num_samples)
        # Naive rolling distance
        ds *= 1.1
        angles = angle_rv.rvs(size=num_samples)
        xs, ys = to_cartesian(ds, angles)
        xs += start_x
        ys += start_y

        H, _, _ = np.histogram2d(
            xs,
            ys,
            # Transform edge bins to inf to capture everything falling off the map
            [
                np.concatenate([[-np.inf], self.x_bins[1:], [np.inf]]),
                np.concatenate([[-np.inf], self.y_bins[1:], [np.inf]]),
            ],
        )
        return H

    def transition_histogram(self, start_xi, start_yi, skill, distance, angle, is_sand):
        start_x = self.x_bins[start_xi] + 0.5 * self.x_tick
        start_y = self.y_bins[start_yi] + 0.5 * self.y_tick
        H = self.sample_shots(start_x, start_y, skill, distance, angle, is_sand)

        transition = np.zeros(self.num_states)

        return_shots = 0
        for xi in range(self.total_x_bins):
            for yi in range(self.total_y_bins):
                samples_in_bin = H.T[yi][xi]
                p_land = self.percent_land[yi][xi]
                p_sand = self.percent_sand[yi][xi]

                grounded_samples = p_land * samples_in_bin
                drowned_samples = (1 - p_land) * samples_in_bin
                return_shots += drowned_samples

                sandy_samples = p_sand * grounded_samples
                green_samples = (1 - p_sand) * grounded_samples
                sand_state = (xi, yi, "sand")
                green_state = (xi, yi, "green")
                if sand_state in self.S_index:
                    transition[self.S_index[sand_state]] += sandy_samples

                if green_state in self.S_index:
                    transition[self.S_index[green_state]] += green_samples

        # Allocate returned shots to the start state
        start_key = (start_xi, start_yi, "sand" if is_sand else "green")
        start_i = self.S_index[start_key]
        transition[start_i] += return_shots

        # Normalize to get probabilities
        transition = transition / max(np.sum(transition), 1)

        return transition

    def transition_for_action_at_state(self, action, state):
        distance, angle = action
        yi, xi, terrain = state
        # This should be a no-op since there should be no non-land states
        if (
            # Not a no-op because we need a dead state
            state == (None, None, None)
            # No-ops, can probably be deleted
            or not self.has_land[yi][xi]
            or (terrain == "green" and self.percent_sand[yi][xi] == 1)
            or (terrain == "sand" and self.percent_sand[yi][xi] == 0)
        ):
            return self.unreachable_transition

        if terrain == "green":
            return self.transition_histogram(
                xi, yi, self.skill, distance, angle, False
            ).flatten()
        elif distance <= (200 + self.skill) / 2:
            return self.transition_histogram(
                xi, yi, self.skill, distance, angle, True
            ).flatten()
        else:
            # In sand, max distance is halved, so treat these actions as invalid
            return self.unreachable_transition

    def gen_action_transitions(self, action):
        return [self.transition_for_action_at_state(action, state) for state in self.S]

    def gen_T_parallel(self):
        print("Generating T...")
        t_start = perf_counter()
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        T = np.array(pool.map(self.gen_action_transitions, self.A))
        # T = np.array([gen_action_transitions(action) for action in A])
        t_end = perf_counter()
        print("T generated in", t_end - t_start, "seconds")
        return T

    def __init__(
        self,
        skill: int,
        rng: np.random.Generator,
        logger: logging.Logger,
        golf_map: sympy.Polygon,
        start: sympy.geometry.Point2D,
        target: sympy.geometry.Point2D,
        sand_traps: list[sympy.Polygon],
        map_path: str,
        precomp_dir: str,
    ) -> None:
        self.skill = skill
        self.rng = rng
        self.logger = logger

        self.target = target.coordinates
        self.green_poly = sympy_poly_to_shapely(golf_map)
        self.sand_polys = [sympy_poly_to_shapely(trap) for trap in sand_traps]

        self.init_geometry()
        self.init_terrain()
        self.init_mdp()

        self.solve_mdp()

    def init_geometry(self):
        # =============================
        # Map quantization fundamentals
        # =============================
        #
        map_min_x, map_min_y, map_max_x, map_max_y = self.green_poly.bounds

        print(f"Map boundaries: x {map_min_x}, {map_max_x} y {map_min_y} {map_max_y}")

        self.x_tick = (map_max_x - map_min_x) / x_quant
        self.y_tick = (map_max_y - map_min_y) / y_quant

        min_x = map_min_x - self.x_tick
        max_x = map_max_x + self.x_tick
        min_y = map_min_y - self.y_tick
        max_y = map_max_y + self.y_tick

        print(f"Bin boundaries: x {min_x}, {max_x} y {min_y} {max_y}")

        self.x_bins = np.linspace(min_x, max_x, x_quant + 2, endpoint=False)
        self.y_bins = np.linspace(min_y, max_y, y_quant + 2, endpoint=False)
        self.total_x_bins = len(self.x_bins)
        self.total_y_bins = len(self.y_bins)
        print("bins:", self.total_x_bins, self.total_y_bins)

    def init_terrain(self):
        # ==============
        # Geometry setup
        # ==============
        #
        cell_polys = [
            [
                ShapelyPolygon(
                    [
                        (x_bin, y_bin),
                        (x_bin + self.x_tick, y_bin),
                        (x_bin + self.x_tick, y_bin + self.y_tick),
                        (x_bin, y_bin + self.y_tick),
                        (x_bin, y_bin),
                    ]
                )
                for x_bin in self.x_bins
            ]
            for y_bin in self.y_bins
        ]

        # =============
        # Terrain types
        # =============
        #
        self.is_land = [
            [self.green_poly.contains(cell_poly) for cell_poly in row]
            for row in cell_polys
        ]
        self.has_land = [
            [self.green_poly.intersects(cell_poly) for cell_poly in row]
            for row in cell_polys
        ]
        self.percent_land = [
            [
                min(self.green_poly.intersection(cell_poly).area / cell_poly.area, 1)
                for cell_poly in row
            ]
            for row in cell_polys
        ]
        self.percent_sand = [
            [
                min(
                    sum(
                        sand_poly.intersection(cell_poly).area
                        for sand_poly in self.sand_polys
                    )
                    / cell_poly.area,
                    1,
                )
                for cell_poly in row
            ]
            for row in cell_polys
        ]

    def init_mdp(self):
        # =====================
        # MDP: States & Actions
        # =====================
        #
        # State setup
        landy_bins = list(
            (yi, xi)
            for yi, xi in product(range(self.total_y_bins), range(self.total_x_bins))
            if self.has_land[yi][xi]
        )
        self.S = list(
            (yi, xi, terrain)
            for ((yi, xi), terrain) in product(landy_bins, ["green", "sand"])
            if (
                (terrain == "green" and self.percent_sand[yi][xi] < 1)
                or (terrain == "sand" and self.percent_sand[yi][xi] > 0)
            )
        )
        # Dead state for invalid moves
        self.S.append((None, None, None))

        # (xi, yi) -> index in S
        self.S_index = {
            (xi, yi, terrain): index for index, (yi, xi, terrain) in enumerate(self.S)
        }

        # Action setup
        distance_levels = np.linspace(1, 200 + self.skill, dist_quant)
        angle_levels = np.linspace(0, 2 * np.pi, angle_quant)
        self.A = list(product(distance_levels, angle_levels))

        self.num_states = len(self.S)
        self.num_actions = len(self.A)
        print("states:", self.num_states)
        print("actions:", self.num_actions)

        # Used for invalid actions
        # S[-1] is an "invalid" sink state: shots going there stay there forever
        self.unreachable_transition = np.array([0 for _ in range(self.num_states)])
        self.unreachable_transition[-1] = 1

        # The expensive step
        self.T = self.gen_T_parallel()

        # =============
        # Reward vector
        # =============
        #
        # All rewards are -1 (penalty for taking another shot) except for the target
        # which has the only positive reward, incentivizing the MDP solving algorithm
        # to find quick paths to reach the target.
        self.R = np.array([-1 for _ in range(self.num_states)])
        target_xi, target_yi = self.to_bin_index(*self.target)
        ti = self.S_index[(target_xi, target_yi, "green")]
        self.R[ti] = 1

    def solve_mdp(self):
        # ===========
        # Train model
        # ===========
        #
        print("Training model...")
        self.mdp = mdptoolbox.mdp.PolicyIteration(self.T, self.R, 0.89, max_iter=20)
        self.mdp.setVerbose()
        self.mdp.run()
        print("Converged in", self.mdp.time)

    def play(
        self,
        score: int,
        golf_map: sympy.Polygon,
        target: sympy.geometry.Point2D,
        sand_traps: List[sympy.geometry.Point2D],
        curr_loc: sympy.geometry.Point2D,
        prev_loc: sympy.geometry.Point2D,
        prev_landing_point: sympy.geometry.Point2D,
        prev_admissible: bool,
    ) -> Tuple[float, float]:
        curr_x = float(curr_loc.x)
        curr_y = float(curr_loc.y)
        in_sand = any(
            trap.contains(ShapelyPoint(curr_x, curr_y)) for trap in self.sand_polys
        )
        xi, yi = self.to_bin_index(curr_x, curr_y)
        curr_bin = self.S_index[(xi, yi, "sand" if in_sand else "green")]
        policy = self.mdp.policy[curr_bin]
        distance, angle = self.A[policy]
        return (distance, angle)