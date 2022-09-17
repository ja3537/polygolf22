import numpy as np
import sympy
import logging
from typing import Tuple, List
from shapely.geometry import Polygon, Point, LineString
from shapely.validation import make_valid
import skgeom as sg
from skgeom.draw import draw
import matplotlib.pyplot as plt
import math
from typing import Tuple
from collections import defaultdict
import time
from scipy.spatial import distance

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


    def construct_edges(self, curr_loc, target, only_construct_from_source=False):
        """Function which creates edges for every node with each other under the following conditions:
            - distance between two nodes < skill_dist_range
            - if the node is <20m from target, there cannot be water in the way.

        Args:
            curr_loc (sympy.geometry.Point2D): Current location
            target (sympy.geometry.Point2D): Target location
        """

        """Graph Creation: Edges
        - In short, we construct directional Edge e: (n1, n2) if our skill level allows us to reach n2 from n1
        - For edges going from:
            - the Node containing the current position:
            use the exact coordinate for the current position as the origin of our circular range
            - a Node that doesn’t contain the current position:
            use the midpoint of that Node (not the midpoint of some unit grid within the Node) as the origin of our
            circular range
        """
        since = time.time()
        source_completed = False
        skill_dist_range = 200 + self.skill
        epsilon = 0.01

        # 2. Connect every node
        for from_node in self.graph.keys():
            # constructing an Edge from curr_loc to another non-curr_loc Node
            if from_node == 'curr_loc':
                # clear existing adjacency list of this from_node
                self.graph[from_node] = []

                for to_node in self.graph.keys():
                    if to_node == from_node:  # 'curr_loc' can't have an Edge with itself
                        continue

                    if to_node == (float(target.x), float(target.y)):
                        if self._euc_dist((int(curr_loc.x), int(curr_loc.y)), to_node) <= 20:
                            line = LineString([(int(curr_loc.x), int(curr_loc.y)), to_node])
                            # i. If yes, calculate bank_distance for this edge
                            if self.shapely_edges.intersects(line):
                                continue
                            else:
                                risk = self.calculate_risk((curr_loc[0], curr_loc[1]), to_node)
                                self.graph[from_node].append([to_node, risk])

                    elif self._euc_dist((int(curr_loc.x), int(curr_loc.y)), to_node) <= skill_dist_range:
                        risk = self.calculate_risk((curr_loc[0], curr_loc[1]), to_node)
                        self.graph[from_node].append([to_node, risk])

                source_completed = True

            # constructing an Edge from a non-curr_loc Node to another non-curr_loc Node
            else:
                if only_construct_from_source and source_completed:  # if only constructing from source, skip this part
                    break

                # clear existing adjacency list of this from_node
                self.graph[from_node] = []

                for to_node in self.graph.keys():
                    # never treat 'curr_loc' as a destination Node; from_node and to_node need to be different
                    if to_node == 'curr_loc' or to_node == from_node:
                        continue

                    if to_node == (float(target.x), float(target.y)):
                        if self._euc_dist(from_node, to_node) <= 20:
                            line = LineString([from_node, to_node])
                            # i. If yes, calculate bank_distance for this edge
                            if self.shapely_edges.intersects(line):
                                continue
                            else:
                                risk = self.calculate_risk(from_node, to_node)
                                self.graph[from_node].append([to_node, risk])

                    elif self._euc_dist(from_node, to_node) <= skill_dist_range:
                        risk = self.calculate_risk(from_node, to_node)
                        self.graph[from_node].append([to_node, risk])

        if DEBUG_MSG:
            print("time for construct_edges:", time.time() - since)

    def calculate_risk(self, start, end):
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        distance = self._euc_dist(start, end)

        dist_deviation = distance/self.skill
        angle_deviation = 2/(2*self.skill)

        max_dist = (distance + dist_deviation)
        min_dist = (distance - dist_deviation)/1.1
        max_angle = angle + angle_deviation
        min_angle = angle - angle_deviation

        #p1 = sg.Point2(start[0]+(max_dist)*math.cos(angle), start[1]+(max_dist)*math.sin(angle))
        p1 = Point(start[0]+(max_dist)*math.cos(angle), start[1]+(max_dist)*math.sin(angle))

        #p4 = sg.Point2(start[0]+(min_dist)*math.cos(angle), start[1]+(min_dist)*math.sin(angle))
        p4 = Point(start[0]+(min_dist)*math.cos(angle), start[1]+(min_dist)*math.sin(angle))

        #p2 = sg.Point2(start[0]+(max_dist)*math.cos(max_angle), start[1]+(max_dist)*math.sin(max_angle))
        p2 = Point(start[0]+(max_dist)*math.cos(max_angle), start[1]+(max_dist)*math.sin(max_angle))

        #p6 = sg.Point2(start[0]+(max_dist)*math.cos(min_angle), start[1]+(max_dist)*math.sin(min_angle))
        p6 = Point(start[0]+(max_dist)*math.cos(min_angle), start[1]+(max_dist)*math.sin(min_angle))

        #p3 = sg.Point2(start[0]+(min_dist)*math.cos(max_angle), start[1]+(min_dist)*math.sin(max_angle))
        p3 = Point(start[0]+(min_dist)*math.cos(max_angle), start[1]+(min_dist)*math.sin(max_angle))

        #p5 = sg.Point2(start[0]+(min_dist)*math.cos(min_angle), start[1]+(min_dist)*math.sin(min_angle))
        p5 = Point(start[0]+(min_dist)*math.cos(min_angle), start[1]+(min_dist)*math.sin(min_angle))
        points = [p1]
        if p2 not in points:
            points.append(p2)
        if p3 not in points:
            points.append(p3)
        if p4 not in points:
            points.append(p4)
        if p5 not in points:
            points.append(p5)
        if p6 not in points:
            points.append(p6)

        #if p1 in [p2, p3, p4, p5, p6] or p2 in [p3, p4, p5, p6] or p3 in [p4, p5, p6] or p4 in [p5, p6] or p5 == p6:
        #    return 1
        if len(points) < 3:
            return 1
        #print(len(points))
        #cone = sg.Polygon(points)
        cone = Polygon(points)
        cone = make_valid(cone)
        #cone_area = cone.area()
        cone_area = cone.area

        #intersect = sg.boolean_set.intersect(cone, self.scikit_poly)
        intersect = self.shapely_poly.intersection(cone).area

        if cone_area == 0:
            return 1
        return intersect/cone_area


    def in_sand_trap(self, x, y):
        point = Point(x, y)
        for trap in self.sand_traps:
            if trap.contains(point):
                return True
        return False

    def get_heuristic(self, x, y):
        point = Point(x, y)
        dist = distance.euclidean(point, self.end_pt)

        max_dist = 200 + self.skill
        heuristic = 0
        if self.in_sand_trap(point):
            heuristic = ((dist - max_dist / 2) / max_dist) + 1
        else:
            heuristic = dist / max_dist

        print("Euc Dist: {}, Heuristic: {}".format(dist, heuristic))
        return heuristic




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

    
        self.get_heuristic(curr_loc[0], curr_loc[1])
        self.get_heuristic(400, 400)


        for trap in self.sand_traps:
            print("Trap : ")
            print(trap)

