import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
import constants
from time import time
from shapely.geometry import Polygon, Point, LineString
from math import pi, atan2, inf, sqrt
import heapq
from dijkstar import Graph, find_path

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

        self.np_curr_loc = np.empty(2)
        self.np_target = np.empty(2)

        self.target = None
        self.current_loc = None

        self.in_polygon = None
        self.origin = None

        self.shapely_polygon = None

        self.n_distances = 25
        self.n_angles = 45

        self.angle_offset = pi

        self.perc_of_path_comp = 0

    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D, prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
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
            Tuple[float, float]: Return a tuple of (landing, doesn't include roll) distance and angle in radians to play the shot
        """

        self.current_loc = np.array([float(curr_loc.x), float(curr_loc.y)])
        # init golf map polygon
        if score == 1:
            self.target = np.array([float(target.x), float(target.y)])
            self.shapely_polygon = Polygon([(p.x,p.y) for p in golf_map.vertices])

            x,y = list(zip(*golf_map.vertices))
            self.origin = (int(min(x)),int(min(y)))
            self.in_polygon = [[False]*int(max(y)-self.origin[1]+1) for _ in range(int(max(x)-self.origin[0]+1))]

            for i in range(len(self.in_polygon)):
                for j in range(len(self.in_polygon[0])):
                    px = int(self.origin[0]+i)
                    py = int(self.origin[1]+j)
                    self.in_polygon[i][j] = self.shapely_polygon.contains(Point(px, py))
            
            self.map = golf_map
            path = self.get_path()

            self.path = LineString([n for n in path.nodes])
                    
        np.copyto(self.np_curr_loc, curr_loc.coordinates, casting='unsafe')
        np.copyto(self.np_target, target.coordinates, casting='unsafe')

        required_dist = np.linalg.norm(self.np_target - self.np_curr_loc)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0
        max_distance = min(200+self.skill, required_dist/roll_factor * 1.50)
        target_angle = atan2(target.y - curr_loc.y, target.x - curr_loc.x)

        best_shot = (0,0)
        max_metric = 0
        best_perc_of_path = 0

        self.perc_of_path_comp = self.percentage_completed(Point(curr_loc.x,curr_loc.y))

        for distance in np.linspace(max_distance, max_distance / self.n_distances, num=self.n_distances):
            for angle in np.linspace(target_angle-self.angle_offset, target_angle+self.angle_offset, num=self.n_angles):
                
                conf = self.est_shot_conf(distance, angle)
                
                p = self.np_curr_loc + distance * np.array([np.cos(angle), np.sin(angle)])
                
                metric, perc_of_path = self.compute_metric(Point(p), conf)

                if metric > max_metric:
                    max_metric = metric
                    best_shot = (distance, angle)
                    best_perc_of_path = perc_of_path

        distance = (required_dist/roll_factor)
        if distance < (200+self.skill)*0.75:
            if distance < 20:
                distance *= 1.10
            else:
                distance *= 1.05
            
            conf = self.est_shot_conf(distance, target_angle)
            metric, perc_of_path = self.compute_metric(Point(target.x,target.y), conf)

            if metric > max_metric:
                max_metric = metric
                best_shot = (distance, target_angle)
                best_perc_of_path = perc_of_path

        return best_shot

    def percentage_completed(self, curr_point):
        line = self.path

        # calculate percentage of path completed
        _, last = line.boundary
        dist_traveled_path = line.project(curr_point)
        total_dist_path = line.project(last)
        perc_of_path_comp = dist_traveled_path / total_dist_path

        return perc_of_path_comp

    def compute_metric(self, curr_point, conf):
        """
            curr_point: Shapely.Point
            conf: float between 0 and 1
        """
        line = self.path

        # calculate distance to our path
        dist_to_line = curr_point.distance(line)

        # calculate percentage of path completed
        perc_of_path_comp = self.percentage_completed(curr_point)

        k1, k2, k3 = 10, 100, 1e-7

        if self.perc_of_path_comp > 0.999:
            return (k1 * conf) / (k3 * dist_to_line+0.00001), perc_of_path_comp

        return (k1 * conf) * (k2 * (perc_of_path_comp - self.perc_of_path_comp)) / (k3 * dist_to_line+0.00001), perc_of_path_comp

    def est_shot_conf(self, distance: float, angle: float, n_tries: int = 50):
        start_time = time()
        n_valid = 0

        # create memory now to save time creating lots of new np arrays
        landing_point = np.empty(2)
        final_point = np.empty(2)
        rot = np.empty(2)
        temp = np.empty(2)

        if constants.min_putter_dist <= distance <= constants.max_dist + self.skill: # normal shot
            case = 0
        elif distance < constants.min_putter_dist: # putter
            case = 1
        else: # invalid, doesn't move
            case = 2

        for _ in range(n_tries):
            
            actual_distance = self.rng.normal(distance, distance/self.skill)
            actual_angle = self.rng.normal(angle, 1/(2*self.skill))

            if case == 0: # normal shot
                rot[:2] = np.cos(actual_angle), np.sin(actual_angle)

                np.multiply(rot, actual_distance, out=temp)
                np.add(self.np_curr_loc, temp, out=landing_point)

                np.multiply(rot, (1.0 + constants.extra_roll)*actual_distance, out=temp)
                np.add(self.np_curr_loc, temp, out=final_point)

            elif case == 1: # putter
                np.copyto(landing_point, self.np_curr_loc)

                rot[:2] = np.cos(actual_angle), np.sin(actual_angle)
                np.multiply(rot, actual_distance, out=temp)
                np.add(self.np_curr_loc, temp, out=final_point)

            else: # invalid, doesn't move
                np.copyto(landing_point, self.np_curr_loc)
                np.copyto(final_point, self.np_curr_loc)

            if self.line_segment_in_polygon(landing_point, final_point, n_points_on_seg=4):
                n_valid += 1

        # t = time() - start_time
        # print('est_shot_conf time:', t)
        return n_valid / n_tries

    # Approx line-seg polygon intersection by checking points along the segment are inside polygon
    def line_segment_in_polygon(self, p1, p2, n_points_on_seg, exact=False):
        direction = np.empty(2)
        temp = np.empty(2)
        p = np.empty(2)

        np.subtract(p2, p1, out=direction)
        for i in np.linspace(0, 1, num=n_points_on_seg):
            np.multiply(i, direction, out=temp)
            np.add(p1, temp, out=p)
            if (exact and not self.shapely_polygon.contains(Point(p[0], p[1]))) or (not exact and not self.point_in_polygon(p)):
                return False
        return True

    def point_in_polygon(self, p):
        x,y = int(p[0])-self.origin[0],int(p[1])-self.origin[1]
        return 0<=x<len(self.in_polygon) and 0<=y<len(self.in_polygon[0]) and self.in_polygon[x][y]

    def compute_graph_nodes(self, golf_map: sympy.Polygon):
        '''
        returns a list of nodes. Each node is a tuple (x,y).
        '''

        nodes = [(float(p.x),float(p.y)) for p in golf_map.vertices]
        a = np.empty(2)
        b = np.empty(2)

        n_vert = len(golf_map.vertices)

        for i in range(n_vert):
            j = (i+1) % n_vert

            np.copyto(a,golf_map.vertices[i], casting='unsafe')
            np.copyto(b,golf_map.vertices[j], casting='unsafe')

            # points along edges
            n_points_on_edge = 5
            for k in range(1,n_points_on_edge+1):
                t = k / (n_points_on_edge+1)
                p = (1-t)*a + t*b
                nodes.append(tuple(p))

            # points along diagonals
            for j in range(i+2, n_vert):

                np.copyto(b,golf_map.vertices[j], casting='unsafe')

                # not sure if endpoints will mess things up so ignore them 
                t = 0.975
                p1 = t*a + (1-t)*b
                p2 = (1-t)*a + t*b

                if self.line_segment_in_polygon(p1, p2, n_points_on_seg=10, exact=True):
                    n_points_on_diag = 2
                    for k in range(1,n_points_on_diag+1):
                        t = k / (n_points_on_diag+1)
                        p = (1-t)*a + t*b
                        nodes.append(tuple(p))
        return nodes

    def get_path(self):
        self.nodes = self.compute_graph_nodes(golf_map=self.map)
        self.nodes.append((self.target[0], self.target[1])) # add target to nodes
        self.nodes.append((self.current_loc[0], self.current_loc[1]))

        max_dist = (200 + self.skill)**2
        def distance(node1,node2):
            val = (node1[0]-node2[0])**2 + (node1[1]-node2[1])**2
            if val > max_dist:
                return float('inf')
            return sqrt(val)
            
        #find shortest path between nodes
        graph = Graph()

        for i in range(len(self.nodes)):
            for j in range(i+1,len(self.nodes)):
                d = distance(self.nodes[i],self.nodes[j])
                if d <= 20 and not self.line_segment_in_polygon(self.nodes[i],self.nodes[j], n_points_on_seg=5):
                    continue
                graph.add_edge(self.nodes[i], self.nodes[j], d)
                graph.add_edge(self.nodes[j], self.nodes[i], d)

        path = find_path(graph, (self.current_loc[0], self.current_loc[1]), (self.target[0], self.target[1]))
        return path
