import numpy as np
import sympy
import logging
from typing import Tuple, List
from shapely.geometry import Polygon, Point, LineString
import skgeom as sg
from skgeom.draw import draw
import matplotlib.pyplot as plt
import math
from typing import Tuple
from collections import defaultdict
import time

DEBUG_MSG = False

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



        #skeleton and critical points
        self.graph = {}  # self.graph[node_i] contains a list of edges where each edge_j = (node_j, heuristic, f_count)
        # self.all_nodes_center = {}
        self.critical_pts = []
        self.scikit_poly = sg.Polygon([(p.x, p.y) for p in golf_map.vertices])
        skel = sg.skeleton.create_interior_straight_skeleton(self.scikit_poly)
        self.draw_skeleton(self.scikit_poly, skel)
        self.construct_nodes(target)
        if DEBUG_MSG:
            draw(self.scikit_poly)
            for v in self.graph.keys():
                plt.plot(v[0], v[1], 'bo')
            plt.savefig('skel.png')

        self.construct_land_bridges(start)
        if DEBUG_MSG:
            draw(self.scikit_poly)
            for v in self.graph.keys():
                plt.plot(v[0], v[1], 'bo')
            plt.savefig('land.png')

        self.construct_more_nodes(start)
        if DEBUG_MSG:
            draw(self.scikit_poly)
            for v in self.graph.keys():
                plt.plot(v[0], v[1], 'bo')
            plt.savefig('more.png')

        if self.needs_edge_init:
            self.construct_edges(start, target, only_construct_from_source=False)
            self.needs_edge_init = False

    def draw_skeleton(self, polygon, skeleton, show_time=False):
        draw(polygon)
        self.critical_pts = []
        for v in skeleton.vertices:
            if v.point not in polygon.vertices:
                self.critical_pts.append([float(v.point.x()), float(v.point.y())])

        out_count = 0
        for point in self.critical_pts:
            if not self.shapely_poly.contains(Point(point[0], point[1])):
                out_count += 1
        # print("out count: ", str(out_count))
        if out_count:
            skel = sg.skeleton.create_exterior_straight_skeleton(self.scikit_poly, 0.1)
            self.draw_skeleton(self.scikit_poly, skel)

    def validate_node(self, x, y, step):
        """ Function which determines if a node of size step x step centered at (x, y) is a valid node in our
        self.shapely_poly

        Args:
            x (float): x-coordinate of node
            y (float): y-coordinate of node
            step (float): size of node

        Returns:
            Boolean: True if node is valid in our map
         """

        # 1. Node center must be inside graph
        valid_edge = 0
        if self.shapely_poly.contains(Point(x, y)):
            # 2. 7/8 points on edge of node must be in graph (we'll count as 8/9, including center)
            for i in np.arange(y - (step / 2), y + step, step / 2):
                for j in np.arange(x - (step / 2), x + step, step / 2):
                    if self.shapely_poly.contains(Point(j, i)):
                        valid_edge += 1
            # return True
        if valid_edge >= 8:
            return True
        else:
            return False

    def construct_nodes(self, target):
        """Function which creates a graph on self.graph with critical points, curr_loc, and target

        Args:
            target (sympy.geometry.Point2D): Target location
        """
        since = time.time()
        self.graph = {'curr_loc': []}

        for point in self.critical_pts:
            if self.shapely_poly.contains(Point(point[0], point[1])):
                self.graph[(point[0], point[1])] = []
        # add target point as node
        self.graph[(float(target.x), float(target.y))] = []

        if DEBUG_MSG:
            print("time for construct_nodes:", time.time() - since)

    def helper_construct_land_bridges(self, from_node, to_node,skill_dist_range,skill_stops,new_nodes):
        distance = self._euc_dist(from_node, to_node)
        line = LineString([from_node, to_node])

        if distance >= skill_dist_range and self.shapely_edges.intersects(line) is False:
            num_stops = math.floor(distance / skill_dist_range)
            len_stop = distance / (num_stops + skill_stops)

            # delta y
            d_y = to_node[1] - from_node[1]
            # delta x
            d_x = to_node[0] - from_node[0]

            if d_x == 0:
                theta = np.sign(d_y) * math.pi / 2
            else:
                theta = math.atan(d_y / d_x)

            for n in range(num_stops):
                offset_y = (n + 1) * len_stop * math.sin(theta)
                offset_x = (n + 1) * len_stop * math.cos(theta)

                stop_point = (from_node[0] + offset_x, from_node[1] + offset_y)
                if self.shapely_poly.contains(Point(stop_point[0], stop_point[1])):
                    new_nodes.append(stop_point)

    def construct_land_bridges(self, curr_loc):
        since = time.time()
        if len(list(self.shapely_poly.exterior.coords)) < 20:
            skill_dist_range = 50 # with not enough nodes(<20), build a landbridge every 50 meters
            if DEBUG_MSG:
                print("construct_land_bridges :: using constant skill_dist_range of", skill_dist_range)
        else:
            skill_dist_range = 200 + self.skill # with enough nodes, build landbridages with max distance

        skill_stops = 2
        new_nodes = []

        for from_node in self.graph.keys():
            if from_node == 'curr_loc':
                for to_node in self.graph.keys():
                    if to_node != from_node:
                        self.helper_construct_land_bridges(curr_loc,to_node,skill_dist_range,skill_stops,new_nodes)
            else:
                for to_node in self.graph.keys():
                    if to_node != 'curr_loc' and to_node != from_node:
                        self.helper_construct_land_bridges(from_node, to_node,skill_dist_range,skill_stops,new_nodes)

        for node in new_nodes:
            self.graph[node] = []
        if DEBUG_MSG:
            print("# nodes after land: " + str(len(self.graph.keys())))
            print("time for land_bridges:", time.time() - since)

    def helper_construct_more_node(self,from_node,to_node,skill_dist_range,new_nodes):
        distance = self._euc_dist(from_node, to_node)
        line = LineString([from_node, to_node])
        if distance >=  skill_dist_range and self.shapely_edges.intersects(line) is True:
            # this should be a LineString obj with 2n points
            intersection = list(self.shapely_edges.intersection(line).geoms)
            if int(len(intersection) / 2) <= 1:
                for i in range(int(len(intersection) / 2)):
                    inter_0 = list(intersection[2 * i].coords)[0]
                    inter_1 = list(intersection[(2 * i) + 1].coords)[0]
                    bank_distance = self._euc_dist(inter_0, inter_1)
                    if bank_distance <= skill_dist_range:
                        d_y = inter_1[1] - inter_0[1]
                        # delta x
                        d_x = inter_1[0] - inter_0[0]

                        if d_x == 0:
                            theta = np.sign(d_y) * math.pi / 2
                        else:
                            theta = math.atan(d_y / d_x)
                            hyp = distance * 0.2
                            bridge_0_count = 0
                            bridge_1_count = 0
                            while (hyp > distance * 0.01):
                                offset_y = hyp * math.sin(theta)
                                offset_x = hyp * math.cos(theta)

                                bridge_0 = (inter_0[0] - offset_x, inter_0[1] - offset_y)
                                bridge_1 = (inter_1[0] + offset_x, inter_1[1] + offset_y)

                                if self.shapely_poly.contains(Point(bridge_0[0], bridge_0[1])):
                                    new_nodes.append(bridge_0)
                                    bridge_0_count += 1
                                    bridge_dist_0 = self._euc_dist(from_node, bridge_0)
                                    if (bridge_dist_0 > skill_dist_range):
                                        num_stops = math.floor(bridge_dist_0 / skill_dist_range)
                                        len_stop = distance / (num_stops + 1)
                                        for n in range(num_stops):
                                            offset_y = (n + 1) * len_stop * math.sin(theta)
                                            offset_x = (n + 1) * len_stop * math.cos(theta)

                                            stop_point = (from_node[0] + offset_x, from_node[1] + offset_y)
                                            if self.shapely_poly.contains(Point(stop_point[0], stop_point[1])):
                                                new_nodes.append(stop_point)
                                if self.shapely_poly.contains(Point(bridge_1[0], bridge_1[1])):
                                    new_nodes.append(bridge_1)
                                    bridge_1_count += 1
                                    bridge_dist_1 = self._euc_dist(bridge_1, to_node)
                                    if (bridge_dist_1 > skill_dist_range):
                                        num_stops = math.floor(bridge_dist_1 / skill_dist_range)
                                        len_stop = distance / (num_stops + 1)
                                        for n in range(num_stops):
                                            offset_y = (n + 1) * len_stop * math.sin(theta)
                                            offset_x = (n + 1) * len_stop * math.cos(theta)

                                            stop_point = (bridge_1[0] + offset_x, bridge_1[1] + offset_y)
                                            if self.shapely_poly.contains(Point(stop_point[0], stop_point[1])):
                                                new_nodes.append(stop_point)
                                hyp = hyp / 2

    def construct_more_nodes(self, curr_loc):
        """Function that builds 'bridges nodes' nearby polygon edges
                to provide options for crossing water.

                Args:
                    curr_loc (sympy.geometry.Point2D): Current location
                """
        since = time.time()

        skill_dist_range = 200 + self.skill
        """ 
        if (self.skill < 80 and len(list(self.shapely_poly.exterior.coords)) > 20):
            mod = round((0.0013 * pow(self.skill, 2)) - (0.2 * self.skill) + 9.9)
            print("mod: ", str(mod)) """

        new_nodes = []
        for from_node in self.graph.keys():
            if from_node == 'curr_loc':
                for to_node in self.graph.keys():
                    if to_node != from_node:  # 'curr_loc' can't have an Edge with itself
                        self.helper_construct_more_node(curr_loc,to_node,skill_dist_range,new_nodes)
            else:
                for to_node in self.graph.keys():
                    if to_node != 'curr_loc' and to_node != from_node:
                        self.helper_construct_more_node(curr_loc, to_node, skill_dist_range, new_nodes)

        # if (self.skill < 80 and len(list(self.shapely_poly.exterior.coords)) > 20):
        total_nodes = len(new_nodes) + len(self.graph.keys())
        if (len(new_nodes) > 2 * len(self.graph.keys()) and total_nodes > 300):
            mod = round(len(new_nodes) / (300 - len(self.graph.keys())))
            new_nodes = new_nodes[::mod]
        for node in new_nodes:
            self.graph[node] = []
        if DEBUG_MSG:
            print("# nodes after water: " + str(len(self.graph.keys())))
            print("time for additional_nodes:", time.time() - since)






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