import os
import pickle
import numpy as np
import sympy
from shapely.geometry import Polygon, Point, LineString
from shapely.validation import make_valid
from skgeom.draw import draw
import skgeom as sg
import matplotlib.pyplot as plt
import math
import logging
from typing import Tuple
from collections import defaultdict
import time

DEBUG_MSG = False  # enable print messages


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


        
        self.needs_edge_init = True
        # if depends on skill
        precomp_path = os.path.join(precomp_dir, "{}_skill-{}.pkl".format(map_path, skill))
        # # if doesn't depend on skill
        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))
        
        # precompute check
        if os.path.isfile(precomp_path):
            # Getting back the objects:
            with open(precomp_path, "rb") as f:
                #self.obj0, self.obj1, self.obj2 = pickle.load(f)
                self.shapely_poly, self.shapely_edges, self.graph, self.critical_pts = pickle.load(f)
        else:
            # Compute objects to store
            #self.obj0, self.obj1, self.obj2 = _
            
            self.graph = {}  # self.graph[node_i] contains a list of edges where each edge_j = (node_j, weight, f_count)
            # self.all_nodes_center = {}
            self.critical_pts = []
            self.shapely_poly = Polygon([(p.x, p.y) for p in golf_map.vertices])
            self.shapely_edges = LineString(list(self.shapely_poly.exterior.coords))

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

            # Dump the objects
            with open(precomp_path, 'wb') as f:
                pickle.dump([self.shapely_poly, self.shapely_edges, self.graph, self.critical_pts], f)
        

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

    def construct_land_bridges(self, curr_loc):
        since = time.time()

        if len(list(self.shapely_poly.exterior.coords)) < 20:
            skill_dist_range = 50
            if DEBUG_MSG:
                print("construct_land_bridges :: using constant skill_dist_range of", skill_dist_range)
        else:
            skill_dist_range = 200 + self.skill

        skill_stops = 2

        new_nodes = []
        for from_node in self.graph.keys():
            if from_node == 'curr_loc':
                for to_node in self.graph.keys():
                    if to_node == from_node:  # 'curr_loc' can't have an Edge with itself
                        continue
                    distance = self._euc_dist((int(curr_loc.x), int(curr_loc.y)), to_node)
                    line = LineString([(int(curr_loc.x), int(curr_loc.y)), to_node])

                    # a. If edge_len < 200: keep as is, regardless of going over water
                    if distance < skill_dist_range:
                        continue

                    # b. If edge_len > 200: check if it goes over water
                    # via line intersecting map
                    else:
                        # i. If yes, calculate bank_distance for this edge
                        if self.shapely_edges.intersects(line):
                            continue
                        # i. If no, create some land bridges based on distance between nodes
                        else:
                            num_stops = math.floor(distance / skill_dist_range)
                            len_stop = distance / (num_stops + skill_stops)

                            # delta y
                            d_y = to_node[1] - curr_loc.y
                            # delta x
                            d_x = to_node[0] - curr_loc.x
                            
                            if d_x == 0:
                                theta = np.sign(d_y) * math.pi/2
                            else:
                                theta = math.atan(d_y / d_x)
                            
                            for n in range(num_stops):
                                offset_y = (n + 1) * len_stop * math.sin(theta)
                                offset_x = (n + 1) * len_stop * math.cos(theta)

                                stop_point = (curr_loc.x + offset_x, curr_loc.y + offset_y)
                                if self.shapely_poly.contains(Point(stop_point[0], stop_point[1])):
                                    new_nodes.append(stop_point)
            else:
                for to_node in self.graph.keys():
                    if to_node == 'curr_loc' or to_node == from_node:
                        continue
                    distance = self._euc_dist(from_node, to_node)
                    line = LineString([from_node, to_node])
                    # a. If edge_len < 200: keep as is, regardless of going over water
                    if distance < skill_dist_range:
                        continue

                    # b. If edge_len > 200: check if it goes over water
                    # via line intersecting map
                    else:
                        # i. If yes, calculate bank_distance for this edge
                        if self.shapely_edges.intersects(line):
                            continue
                        # i. If no, create some land bridges based on distance between nodes
                        else:
                            num_stops = math.floor(distance / skill_dist_range)
                            len_stop = distance / (num_stops + skill_stops)

                            # delta y
                            d_y = to_node[1] - from_node[1]
                            # delta x
                            d_x = to_node[0] - from_node[0]

                            if d_x == 0:
                                theta = np.sign(d_y) * math.pi/2
                            else:
                                theta = math.atan(d_y / d_x)

                            for n in range(num_stops):
                                offset_y = (n + 1) * len_stop * math.sin(theta)
                                offset_x = (n + 1) * len_stop * math.cos(theta)

                                stop_point = (from_node[0] + offset_x, from_node[1] + offset_y)
                                if self.shapely_poly.contains(Point(stop_point[0], stop_point[1])):
                                    new_nodes.append(stop_point)

        for node in new_nodes:
            self.graph[node] = []
        if DEBUG_MSG:
            print("# nodes after land: " + str(len(self.graph.keys())))
            print("time for land_bridges:", time.time() - since)

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
                    if to_node == from_node:  # 'curr_loc' can't have an Edge with itself
                        continue

                    distance = self._euc_dist((int(curr_loc.x), int(curr_loc.y)), to_node)
                    line = LineString([(int(curr_loc.x), int(curr_loc.y)), to_node])

                    # a. If edge_len < 200: keep as is, regardless of going over water
                    if distance < skill_dist_range:
                        continue

                    # b. If edge_len > 200: check if it goes over water
                    # via line intersecting map
                    else:
                        # i. If yes, calculate bank_distance for this edge
                        if self.shapely_edges.intersects(line):
                            # this should be a LineString obj with 2n points
                            intersection = list(self.shapely_edges.intersection(line).geoms)

                            # if we cross water more than once, ignore path.
                            if int(len(intersection) / 2) > 1:
                                continue

                            for i in range(int(len(intersection) / 2)):
                                inter_0 = list(intersection[2 * i].coords)[0]
                                inter_1 = list(intersection[(2 * i) + 1].coords)[0]
                                bank_distance = self._euc_dist(inter_0, inter_1)

                                # - bank_distance > 200: discard edge
                                if bank_distance > skill_dist_range:
                                    continue
                                # - bank_distance < 200: create two new nodes at the edge of bank 
                                else:
                                    # delta y
                                    d_y = inter_1[1] - inter_0[1]
                                    # delta x
                                    d_x = inter_1[0] - inter_0[0]

                                    if d_x == 0:
                                        theta = np.sign(d_y) * math.pi / 2
                                    else:
                                        theta = math.atan(d_y / d_x)

                                    hyp = distance*0.2
                                    bridge_0_count = 0
                                    bridge_1_count = 0
                                    while (hyp > distance*0.01):
                                        
                                        offset_y = hyp * math.sin(theta)
                                        offset_x = hyp * math.cos(theta)

                                        bridge_0 = (inter_0[0] - offset_x, inter_0[1] - offset_y)
                                        bridge_1 = (inter_1[0] + offset_x, inter_1[1] + offset_y)
                                        if self.shapely_poly.contains(Point(bridge_0[0], bridge_0[1])):
                                            new_nodes.append(bridge_0)
                                            bridge_0_count += 1
                                            bridge_dist_0 = self._euc_dist((int(curr_loc.x), int(curr_loc.y)), bridge_0)
                                            if (bridge_dist_0 > skill_dist_range):
                                                num_stops = math.floor(bridge_dist_0 / skill_dist_range)
                                                len_stop = distance / (num_stops + 1)
                                                for n in range(num_stops):
                                                    offset_y = (n + 1) * len_stop * math.sin(theta)
                                                    offset_x = (n + 1) * len_stop * math.cos(theta)

                                                    stop_point = (curr_loc.x + offset_x, curr_loc.y + offset_y)
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
                                        
            else:
                for to_node in self.graph.keys():
                    if to_node == 'curr_loc' or to_node == from_node:
                        continue
                    distance = self._euc_dist(from_node, to_node)
                    line = LineString([from_node, to_node])
                    # a. If edge_len < 200: keep as is, regardless of going over water
                    if distance < skill_dist_range:
                        continue

                    # b. If edge_len > 200: check if it goes over water
                    # via line intersecting map
                    else:
                        # i. If yes, calculate bank_distance for this edge
                        if self.shapely_edges.intersects(line):
                            intersection = list(self.shapely_edges.intersection(line).geoms)

                            # test: if we cross water more than once, ignore path.
                            if int(len(intersection) / 2) > 1:
                                continue
                            
                            for i in range(int(len(intersection) / 2)):
                                inter_0 = list(intersection[2 * i].coords)[0]
                                inter_1 = list(intersection[(2 * i) + 1].coords)[0]
                                bank_distance = self._euc_dist(inter_0, inter_1)

                                # - bank_distance > 200: discard edge
                                if bank_distance > skill_dist_range:
                                    continue
                                # - bank_distance < 200: create two new nodes at the edge of bank 
                                else:
                                    # delta y
                                    d_y = inter_1[1] - inter_0[1]
                                    # delta x
                                    d_x = inter_1[0] - inter_0[0]

                                    if d_x == 0:
                                        theta = np.sign(d_y) * math.pi / 2
                                    else:
                                        theta = math.atan(d_y / d_x)

                                    hyp = distance*0.2
                                    bridge_0_count = 0
                                    bridge_1_count = 0
                                    while (hyp > distance*0.01):
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
        #if (self.skill < 80 and len(list(self.shapely_poly.exterior.coords)) > 20):
        total_nodes = len(new_nodes) + len(self.graph.keys())
        if (len(new_nodes) > 2 * len(self.graph.keys()) and total_nodes > 300):
            mod = round(len(new_nodes)/(300 - len(self.graph.keys())))
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

    @staticmethod
    def _euc_dist(pt1, pt2):
        pt1 = np.array((float(pt1[0]), float(pt1[1])))
        pt2 = np.array((float(pt2[0]), float(pt2[1])))
        return np.linalg.norm(pt1 - pt2)

    # @staticmethod
    # def _get_node_center(unit_centers):
    #     if len(unit_centers) % 2:  # if number of Units is odd, take the middle Unit center as the Node center
    #         node_center = unit_centers[int(len(unit_centers) / 2)]
    #     else:  # if number of Units is even, average the middle two Unit centers as the Node center
    #         mid_left_unit_center = unit_centers[int(len(unit_centers) / 2) - 1]
    #         mid_right_unit_center = unit_centers[int(len(unit_centers) / 2)]
    #         node_center_x = (mid_left_unit_center[0] + mid_right_unit_center[0]) / 2
    #         node_center_y = (mid_left_unit_center[1] + mid_right_unit_center[1]) / 2
    #         node_center = (node_center_x, node_center_y)
    #     return node_center

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

        """ final_area = 0
        for poly in intersect:
            final_area += poly.outer_boundary().area()
            for hole in poly.holes:
                final_area -= hole.area()
         """
        #print("Calculated")
        #print(final_area/cone_area)
        #return final_area/cone_area
        # TODO
        if cone_area == 0:
            return 1
        return intersect/cone_area
    
    def BFS(self, target, tolerance):
        """Function that performs BFS on the graph of nodes to find a path to the target, prioritizing minimum
        moves in order to minimize our score.
        
        Returns:
            Node to make a move towards.
        """
        since = time.time()

        visited = defaultdict(int)
        # compare_target = ((target.x, target.y),)
        compare_target = (float(target.x), float(target.y))
        queue = []
        queue.append([['curr_loc', 1]])
        final_path = []
        while queue:
            path = queue.pop(0)
            node = path[-1][0]
            if node != 'curr_loc' and node == compare_target:
                # print(path)
                final_path = path
                break

            if 0 < visited[node] <= len(path):
                continue

            visited[node] = len(path)

            adj = self.graph[node]
            for a in adj:
                if a[1] < tolerance:
                    continue
                new_path = list(path)
                new_path.append(a)
                queue.append(new_path)
        if len(final_path) < 2:
            if DEBUG_MSG:
                print("sanity check: ", str(len(final_path)))
                print("time for bfs:", time.time() - since)
            return "default"

        move = final_path[1][0]
        risk = min([a[1] for a in final_path])

        if DEBUG_MSG:
            print("time for bfs:", time.time() - since)
            print("final_path:", [move for move in final_path[1:]])
        return [sympy.geometry.Point2D(move[0], move[1]), len(final_path), risk]

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

        if DEBUG_MSG:
            print("curr_loc", float(curr_loc.x), float(curr_loc.y))

        required_dist = curr_loc.distance(target)
        """ 
        Case 1: required_dist > 20m
            - create a map with 1m-sized nodes

        Case 2: requier_dist < 20m (putting dist)
            - more specific map with .05m-sized nodes 20m around from current point
         """

        #if score == 1:  # turn 1
            
        if score != 1:
            self.construct_edges(curr_loc, target,
                                 only_construct_from_source=True)  # only construct outgoing edges of curr_loc
        max_score = 0
        move = "default"
        for i in range(1, 5):
            tolerance = .25*i
            m = self.BFS(target, tolerance)
            if m == "default":
                continue
            score = m[2]/m[1]
            if DEBUG_MSG:
                print("score")
                print(score)
                print("risk")
                print(m[2])
                print("Path Length")
                print(m[1])
            if score >= max_score:
                max_score = score
                move = m[0]

        roll_factor = 1.1
        if required_dist < 20:
            roll_factor = 1.0

            # consider putting behavior
            """ line = LineString([(curr_loc.x, curr_loc.y), (target.x, target.y)])
            if not self.shapely_edges.intersects(line):
                print("default putt")
                move = "default" """

        if move == "default":
            if DEBUG_MSG:
                print("******default******")
            distance = sympy.Min(200 + self.skill, required_dist / roll_factor)
            angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
            return (distance, angle)

        distance = curr_loc.distance(move)/roll_factor
        angle = sympy.atan2(move.y - curr_loc.y, move.x - curr_loc.x)
        # distance = sympy.Min(200 + self.skill, required_dist / roll_factor)
        # angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        return (distance, angle)
