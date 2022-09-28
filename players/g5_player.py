import logging
import multiprocessing
import os.path
import pickle
from itertools import product
from os import makedirs
from time import perf_counter
from typing import List, Tuple

import mdptoolbox
import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy.stats import norm
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

# =====================
# Problem Configuration
# =====================
#
DEBUG = False
samples = 20
x_quant = 20
y_quant = 20
dist_quant = 20
angle_quant = 24

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


def to_polar(x, y):
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


def debug(*args):
    if DEBUG:
        print(*args)


# ======
# Player
# ======
#
class Player:
    def to_bin_index(self, x, y):
        xi = next(xi for xi, x_bin in enumerate(self.x_bins) if x_bin > x) - 1
        yi = next(yi for yi, y_bin in enumerate(self.y_bins) if y_bin > y) - 1
        return xi, yi

    def in_sand(self, x, y):
        return any(trap.contains(ShapelyPoint(x, y)) for trap in self.sand_polys)

    # ===========================
    # Shot transition calculation
    # ===========================
    #
    def sample_shots(
        self,
        start_x,
        start_y,
        skill,
        distance,
        angle,
        in_sand,
        num_samples=samples,
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
        # ds *= 1.1

        angles = angle_rv.rvs(size=num_samples)
        xs, ys = to_cartesian(ds, angles)
        xs += start_x
        ys += start_y

        # Expensive, better rolling calculation
        roll_mask = np.array(
            [
                0
                if self.in_sand(x, y)
                or not self.green_poly.contains(ShapelyPoint(x, y))
                else 1
                for x, y in zip(xs, ys)
            ]
        )
        ds *= roll_mask * 1.1
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
        debug("Generating T...")
        t_start = perf_counter()

        # In parallel
        # If this code is causing problems in the tournament, comment out the
        # two lines below and uncomment the line below "Serially"
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        T = np.array(pool.map(self.gen_action_transitions, self.A))

        # Serially (performance testing)
        # T = np.array([self.gen_action_transitions(action) for action in self.A])

        t_end = perf_counter()
        debug("T generated in", t_end - t_start, "seconds")
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
        self.map_file = map_path

        self.start = start.coordinates
        self.target = target.coordinates
        self.green_poly = sympy_poly_to_shapely(golf_map)
        self.sand_polys = [sympy_poly_to_shapely(trap) for trap in sand_traps]

        self.init_geometry()
        self.init_terrain()
        self.init_mdp()

        precomp_path = os.path.join(
            precomp_dir, "{}_skill-{}.pkl".format(map_path, skill)
        )

        # precompute check
        if os.path.isfile(precomp_path):
            # Getting back the objects:
            debug("Found cached policy", precomp_path)
            with open(precomp_path, "rb") as f:
                if DEBUG:
                    self.T, self.policy = pickle.load(f)
                else:
                    self.policy = pickle.load(f)
        else:
            self.solve_mdp()
            if DEBUG:
                self.visualize_value()
                self.visualize_policy()

            # Dump the objects
            with open(precomp_path, "wb") as f:
                if DEBUG:
                    pickle.dump([self.T, self.policy], f)
                else:
                    pickle.dump(self.policy, f)

        # if DEBUG:
        #     self.visualize_all_shots(*self.start)

    def init_geometry(self):
        # =============================
        # Map quantization fundamentals
        # =============================
        #
        map_min_x, map_min_y, map_max_x, map_max_y = self.green_poly.bounds

        debug(f"Map boundaries: x {map_min_x}, {map_max_x} y {map_min_y} {map_max_y}")

        self.x_tick = (map_max_x - map_min_x) / x_quant
        self.y_tick = (map_max_y - map_min_y) / y_quant

        min_x = map_min_x - self.x_tick
        max_x = map_max_x + self.x_tick
        min_y = map_min_y - self.y_tick
        max_y = map_max_y + self.y_tick

        debug(f"Bin boundaries: x {min_x}, {max_x} y {min_y} {max_y}")

        self.x_bins = np.linspace(min_x, max_x, x_quant + 2, endpoint=False)
        self.y_bins = np.linspace(min_y, max_y, y_quant + 2, endpoint=False)
        self.total_x_bins = len(self.x_bins)
        self.total_y_bins = len(self.y_bins)
        debug("bins:", self.total_x_bins, self.total_y_bins)

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
        debug("states:", self.num_states)
        debug("actions:", self.num_actions)

        # Used for invalid actions
        # S[-1] is an "invalid" sink state: shots going there stay there forever
        self.unreachable_transition = np.array([0 for _ in range(self.num_states)])
        self.unreachable_transition[-1] = 1

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
        # The expensive step
        self.T = self.gen_T_parallel()

        # ===========
        # Train model
        # ===========
        #
        debug("Training model...")
        self.mdp = mdptoolbox.mdp.PolicyIteration(self.T, self.R, 0.89, max_iter=20)
        if DEBUG:
            self.mdp.setVerbose()
        self.mdp.run()
        debug("Converged in", self.mdp.time)

        self.policy = self.mdp.policy

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
        policy = self.policy[curr_bin]
        planned_distance, planned_angle = self.A[policy]

        # Compensate for difference between planned shot (center of tile) and
        # actual ball location
        cx = self.x_bins[xi] + 0.5 * self.x_tick
        cy = self.y_bins[yi] + 0.5 * self.y_tick

        planned_dx, planned_dy = to_cartesian(planned_distance, planned_angle)
        end_x = cx + planned_dx
        end_y = cy + planned_dy

        distance, angle = to_polar(end_x - curr_x, end_y - curr_y)
        if DEBUG:
            self.logger.info(f"Distance adjusted by {distance - planned_distance}")
            self.logger.info(f"Angle adjusted by {angle - planned_angle}")

        # Putting strategy
        target_x = float(self.target[0])
        target_y = float(self.target[1])
        distance_to_target = np.linalg.norm(
            np.array([curr_x, curr_y]) - np.array([target_x, target_y])
        )
        if not in_sand and distance_to_target < 20:
            # The selection of putting distance is inspired by Fall 2022 Group 8's approach
            angle = np.arctan2(target_y - curr_y, target_x - curr_x)
            if self.skill >= 60:
                overshoot_factor = 1.5
            elif self.skill >= 40:
                overshoot_factor = 1.2
            elif self.skill >= 20:
                overshoot_factor = 1.05
            else:
                if distance_to_target > 10:
                    overshoot_factor = 0.9
                elif distance_to_target > 1:
                    overshoot_factor = 0.95
                else:
                    overshoot_factor = 1.02
            distance = distance_to_target * overshoot_factor
            distance = min(20.0, distance)

            return distance, angle

        if distance > 200 + self.skill:
            if DEBUG:
                self.logger.warning(
                    "Compensated shot is longer than max distance rating"
                )
            # Take a small shot towards the center of the tile
            dx = cx - curr_x
            dy = cy - curr_y
            return to_polar(dx, dy)

        return (distance, angle)

    # =====================
    # Visualization Helpers
    # =====================
    #
    def draw_bins(self):
        for x in self.x_bins:
            plt.axvline(x=x, color="black", alpha=0.1)
        for y in self.y_bins:
            plt.axhline(y=y, color="black", alpha=0.1)

    def draw_map(self):
        plt.fill(
            *list(zip(*self.green_poly.exterior.coords)),
            facecolor="#bbff66",
            edgecolor="black",
            linewidth=1,
        )
        start_x, start_y = self.start
        plt.plot(start_x, start_y, "b.")
        target_x, target_y = self.target
        plt.plot(target_x, target_y, "r.")

        for trap_poly in self.sand_polys:
            plt.fill(
                *list(zip(*trap_poly.exterior.coords)),
                facecolor="#ffffcc",
                edgecolor="black",
                linewidth=1,
            )

    def reset_figure(self):
        plt.clf()
        plt.gca().invert_yaxis()
        # plt.gca().set_aspect(width / height)
        self.draw_bins()
        self.draw_map()

    def overlay_tiles(self, tiles, vmin=None, vmax=None):
        X, Y = np.meshgrid(
            self.x_bins + 0.5 * self.x_tick, self.y_bins + 0.5 * self.y_tick
        )
        plt.pcolormesh(X, Y, tiles, alpha=0.5, vmin=vmin, vmax=vmax)

    # ================
    # Visualize values
    # ================
    #
    def visualize_value(self):
        self.reset_figure()
        v_hist = np.zeros((self.total_y_bins, self.total_x_bins))
        sorted_values = sorted(self.mdp.V[:-1])
        vmin = sorted_values[1]
        vmax = sorted_values[-1]
        for i, value in enumerate(self.mdp.V[:-1]):
            yi, xi, terrain = self.S[i]
            if terrain == "green":
                v_hist[yi][xi] = value
            elif terrain == "sand" and self.percent_sand[yi][xi] > 0.9:
                v_hist[yi][xi] = value
        self.reset_figure()
        self.overlay_tiles(v_hist, vmin=vmin, vmax=vmax)
        plt.title(f"Values: {self.map_file}, skill {self.skill}")
        plt.savefig("value.png", dpi=400)

    # ================
    # Visualize policy
    # ================
    #
    def visualize_policy(self):
        self.reset_figure()
        for i, policy in enumerate(self.mdp.policy[:-1]):
            yi, xi, terrain = self.S[i]
            distance, angle = self.A[policy]
            dx, dy = to_cartesian(distance, angle)
            start_x = self.x_bins[xi] + 0.5 * self.x_tick
            start_y = self.y_bins[yi] + 0.5 * self.y_tick
            plt.arrow(
                start_x,
                start_y,
                dx,
                dy,
                color="black" if terrain == "green" else "red",
                alpha=0.2,
                linewidth=1,
                head_width=8,
                head_length=8,
                length_includes_head=True,
            )
        plt.title(f"Policy: {self.map_file}, skill {self.skill}")
        plt.savefig("policy.png", dpi=400)

    # ===================
    # Visualize all shots
    # ===================
    #
    # Convert into an animation with
    # `convert -delay 0 -loop 0 shots/*.png -quality 95 shots.mp4`
    #
    def visualize_all_shots(self, x, y):
        makedirs("shots", exist_ok=True)
        test_xi, test_yi = self.to_bin_index(x, y)
        test_terrain = "green"

        map_min_x, map_min_y, map_max_x, map_max_y = self.green_poly.bounds

        for ai, action in enumerate(self.A):
            print("Rendering", ai + 1, "out of", len(self.A))

            self.reset_figure()
            plt.gca().set_xlim([map_min_x, map_max_x])
            plt.gca().set_ylim([map_min_y, map_max_y])
            plt.gca().invert_yaxis()

            self.draw_map()
            self.draw_bins()

            state_bin = self.S_index[(test_xi, test_yi, test_terrain)]
            transitions = self.T[ai][state_bin][:-1]
            transition_probs = np.zeros((self.total_y_bins, self.total_x_bins))
            for si, prob in enumerate(transitions):
                yi, xi, _ = self.S[si]
                transition_probs[yi][xi] = prob
            self.overlay_tiles(transition_probs, vmin=0, vmax=1)

            start_x = self.x_bins[test_xi] + 0.5 * self.x_tick
            start_y = self.y_bins[test_yi] + 0.5 * self.y_tick

            # Plot policy vector
            policy = self.policy[state_bin]
            distance, angle = self.A[policy]
            dx, dy = to_cartesian(distance, angle)
            plt.arrow(
                start_x,
                start_y,
                dx,
                dy,
                color="green",
                alpha=0.5,
                linewidth=1,
                head_width=8,
                head_length=8,
                length_includes_head=True,
            )

            # Plot this action
            distance, angle = action
            dx, dy = to_cartesian(distance, angle)
            plt.arrow(
                start_x,
                start_y,
                dx,
                dy,
                color="black",
                alpha=0.5,
                linewidth=1,
                head_width=8,
                head_length=8,
                length_includes_head=True,
            )

            plt.title(f"Distance: {distance}, Angle: {angle}")

            plt.savefig(f"shots/{ai}.png")
