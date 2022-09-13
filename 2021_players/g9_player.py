import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
import shapely.geometry
import random
from matplotlib import pyplot as plt
from collections import deque
import math
from queue import PriorityQueue
import heapq
from heapq import heappush, heappop

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
        
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.quick_map = shapely.geometry.Polygon([(p.x,p.y) for p in golf_map.vertices])
        self.dmap = None
        self.pmap = None
        self.cell_width = 1
        self.rows = None
        self.cols = None
        self.max_distance = 200 + skill -.001
        self.distances = [self.max_distance*0.3, self.max_distance*0.4,self.max_distance*0.5, self.max_distance*0.6,self.max_distance*0.7, self.max_distance*0.8,self.max_distance*0.9, self.max_distance]
        #self.angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 3 * np.pi / 4, 5 * np.pi / 6, np.pi, 7 * np.pi / 6, 5 * np.pi / 4, 4 * np.pi / 3, 3 * np.pi / 2, 5 * np.pi / 3, 7 * np.pi / 4, 11 * np.pi / 6]
        self.angles = []
        i = 0
        num_angles = 20
        while(i < num_angles):
            self.angles.append(i * 2 * np.pi / num_angles)
            i += 1

        minx, miny, maxx, maxy = self.quick_map.bounds
        self.minx = minx
        self.miny = miny

        width = maxx - minx
        height = maxy - miny
        self.cols = int(np.ceil(width / self.cell_width))
        self.rows = int(np.ceil(height / self.cell_width))
        self.zero_center = shapely.geometry.Point(minx + self.cell_width / 2, maxy - self.cell_width / 2)

        precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))
        # precompute check
        if os.path.isfile(precomp_path):
            # Getting back the objects:
            with open(precomp_path, "rb") as f:
                self.dmap = pickle.load(f)
        else:
            # Compute objects to store
            self.precompute()

            # Dump the objects
            with open(precomp_path, 'wb') as f:
                pickle.dump(self.dmap, f)
        
    def get_landing_point(self, curr_loc: shapely.geometry.Point, distance: float, angle: float):
        """
    	    Args:
    	        curr_loc (sympy.geometry.Point2D): current location
    	        distance (float): The distance to next potential landing point
    	        angle (float): The angle coordinate to the next potential landing point
    	    Returns:
    	        the potential landing point as a sympy.Point2D object
    	"""
        return shapely.geometry.Point(curr_loc.x + distance * np.cos(float(angle)), curr_loc.y + distance * np.sin(float(angle)))

    
    def get_center(self, r: int, c: int):
        x = self.zero_center.x + c * self.cell_width
        y = self.zero_center.y - r * self.cell_width
        return shapely.geometry.Point(x, y)

    def get_row_col(self, x, y): #will return row, col closest to point (if within dmap)
        c = round((x - self.zero_center.x)/self.cell_width)
        r = round((self.zero_center.y - y)/self.cell_width)

        return r, c

    def get_corners(self, r: int, c: int):
        center = self.get_center(r,c)
        offset = self.cell_width / 2
        x = center.x
        y = center.y
        upper_left = shapely.geometry.Point(x - offset, y + offset)
        upper_right = shapely.geometry.Point(x + offset, y + offset)
        lower_left = shapely.geometry.Point(x - offset, y - offset)
        lower_right = shapely.geometry.Point(x + offset, y - offset)
        return [upper_left, upper_right, lower_left, lower_right]

    def in_bounds(self, row, col):
        if row >= 0 and col >=0 and row < self.rows and col < self.cols:
            return True
        return False

    def get_neighbors(self, row, col):
        neighbors_list = []
        for i in range(-1,2):
            for j in range(-1,2):
                if i == 0 and j == 0:
                    continue
                neighborRow = row + i
                neighborCol = col + j

                if self.in_bounds(neighborRow, neighborCol):
                    neighbors_list.append((neighborRow, neighborCol))

        return neighbors_list



    def brushfire(self, q):
        while(len(q) != 0):
            row, col, dist, waterPoint = q.popleft()

            neighbors_list = self.get_neighbors(row, col)
            for nRow, nCol in neighbors_list:
                if self.dmap[nRow][nCol] == 1: #is on land
                    nCenter = self.get_center(nRow, nCol)
                    ndist = nCenter.distance(waterPoint) #distance from neighbor to water
                    pValue = self.pmap[nRow, nCol]
                    if (pValue == -1) or (pValue > ndist): #either uninitialized or current distance is smaller
                        self.pmap[nRow, nCol] = ndist
                        q.append((nRow, nCol, ndist, waterPoint))

    def get_polar(self, start, end): #converts from square to polar. Returns r, \theta
        angle = sympy.atan2(end.y - start.y, end.x - start.x)
        distance = start.distance(end)

        return distance, angle



    def p_in_water(self,start, end):
        d = start.distance(end)
        sigma = d/self.skill
        r, c = self.get_row_col(end.x, end.y)
        dw = self.pmap[r, c] #distance to water for end point
        alpha = dw/sigma #number of standard deviations to water for end point
        if 1 - alpha > 0.999:
            alpha = 0.001

        if alpha <= 1:
            p = 1 - (1.848*math.exp(-1/alpha))
            if 1 - p < 0.001:
                return 0.999
            else:
                return p
        elif alpha < 2:
            return 1 - (0.68 + 0.27*(1- alpha))
        else:
            return 0

    def simulate_shot(self, distance, angle): #draws from normal distributions and returns r, theta
        actual_distance = self.rng.normal(distance, distance / self.skill)
        actual_angle = self.rng.normal(angle, 1 / (2 * self.skill))

        return actual_distance, actual_angle

    def on_land(self, points): #returns true if ALL points in the list points is on land
        for p in points:
            r, c = self.get_row_col(p.x, p.y)
            if(self.in_bounds(r, c)):
                if self.dmap[r, c] != 1:
                    return False #in bounds but in water
            else:
                return False #out of bounds of dmap

        return True #all points in bounds and on land

    def p_in_water2(self, start, end):
        d, angle = self.get_polar(start, end)
        land_count = 0
        runs = 50

        for i in range(runs):
            r, theta = self.simulate_shot(d, angle)
            p1 = self.get_landing_point(start, r, theta) #landing point
            p2 = self.get_landing_point(start, 1.05*r, theta) #halfway between
            p3 = self.get_landing_point(start, 1.1*r, theta) #rolling point

            if self.on_land([p1, p2, p3]):
                land_count += 1

        p_water = (runs - land_count)/runs
        if p_water > 0.99:
            p_water = 0.99

        return p_water





    def expected_strokes(self, start, end):
        ratio = self.p_in_water2(start,end)

        return 1 + (ratio)/(1-ratio)

    
    def precompute(self):
        # minx, miny, maxx, maxy = self.quick_map.bounds
        # self.minx = minx
        # self.miny = miny

        # width = maxx - minx
        # height = maxy - miny
        # self.cols = int(np.ceil(width / self.cell_width))
        # self.rows = int(np.ceil(height / self.cell_width))
        # self.zero_center = shapely.geometry.Point(minx + self.cell_width / 2, maxy - self.cell_width / 2)

        self.dmap = np.zeros((self.rows, self.cols), dtype=np.int8)
        # self.pmap = np.zeros((self.rows, self.cols), dtype=np.int8)



        for row in range(self.rows):
            for col in range(self.cols):
                corners = self.get_corners(row, col)
                water = 0
                land = 0
                for point in corners:
                    if self.quick_map.contains(point):
                        land += 1
                    else:
                        water += 1

                if land == 4:
                    # if all four points on land, then set dmap to 1
                    self.dmap[row, col] = 1

                elif water == 4:
                    # if all four points on water, then set dmap to 0
                    self.dmap[row, col] = 0

                else:
                    # in else ==> some points on land, some in water ==> we are on an edge cell
                    self.dmap[row, col] = 0

        ### Commented out code saves precomputed brushfire map for testing purposes
        # if os.path.exists(self.tmp):
        #     with open(self.tmp, 'rb') as f:
        #         self.pmap = np.load(f)
        #         print("loaded from file")
        #         return
        # #self.brushfire(q)
        # with open(self.tmp, 'wb') as f:
        #     np.save(f, self.pmap)


    # Generate potential branches for the A* searching algorithm
    def generate_branches(self, points, displacement_angle):
        pt = points[-1]
        li = []
        for distance in self.distances:
            for angle in self.angles:
                angle = angle + displacement_angle
                if(angle > 2 * np.pi):
                    angle = angle - 2 * np.pi
                new_point = self.get_landing_point(pt, distance, angle)
                valid = True
                for i in range(len(points) - 1):
                    if points[i].distance(new_point) < (self.max_distance/2):
                        valid = False
                        break
                if valid:
                    li.append(new_point)

        return li

    def a_star(self, start, target, curr_loc, target_loc):
        count = 0
        secondCount = 0
        # print("starting a star")

        #pq = PriorityQueue() #convention for data insertion is [heuristic, path, estimated total strokes for this path]

        E = start.distance(target)/(200 + self.skill)
        s = 0 #no shots taken so far
        path = [start]
        #pq.put([E, path, 0])
        heap = [(E, secondCount, 0, path)]
        displacement_angle = sympy.atan2(target_loc.y - curr_loc.y, target_loc.x - curr_loc.x)
        while heap:
            count += 1
            h, _, strokes, path = heappop(heap)

            lastPoint = path[-1]
            if lastPoint.distance(target) <= 200 + self.skill:
                # print("found path")
                # print(count)
                path.append(target)
                return path


            possibleShots = self.generate_branches(path, displacement_angle)


            for shot in possibleShots:
                secondCount += 1

                r,c = self.get_row_col(shot.x, shot.y)
                if(self.in_bounds(r, c)):
                    s = strokes + self.expected_strokes(lastPoint, shot)
                    d = shot.distance(target)
                    E = d/(200 + self.skill)
                    h = s + E #heuristic
                    newPath = path.copy()
                    newPath.append(shot)
                    heappush(heap, (h, secondCount, s, newPath))



    def approach(self, s, t):
        d = s.distance(t)
        r = d/1.1
        angle = float(sympy.atan2(t.y - s.y, t.x - s.x))

        pt = self.get_landing_point(s, r, angle)


        bestProbability = self.expected_strokes(s, pt)
        bestr = r
        bestangle = angle

        if bestProbability < 1.1: #if shooting straight at the hole will go in water less than 1/10th of the time, just shoot
            return r, angle


        for i in range(-20,21,2):
            for j in range(-20,21,2): #will check 100 points in 2 meter increments around aiming directly at hole
                testPoint = shapely.geometry.Point(pt.x + i, pt.y + j)
                testr, testangle = self.get_polar(s, testPoint)
                if(testr > self.max_distance): #break if point too far away
                    break
                strokes = self.expected_strokes(s, testPoint)
                if strokes < bestProbability:
                    bestProbability = strokes
                    r = testr
                    angle = testangle

        return r, angle




    
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
            Tuple[float, float]: Return a tuple of distance and angle in radians to play the shot
        """

        # Testing
        count = 0
        # .bounds function can get (minx, miny, maxx, maxy) tuple (float values) that bounds the object

        s = shapely.geometry.Point(curr_loc.x, curr_loc.y)
        t = shapely.geometry.Point(target.x, target.y)

        # Within goal
        if s.distance(t) <= self.max_distance:
            angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
            if s.distance(t) < 20:
                distance = s.distance(t)
            else:
                distance, angle = self.approach(s,t)
            return (distance, angle)
        else:
            s = shapely.geometry.Point(curr_loc.x, curr_loc.y)
            t = shapely.geometry.Point(target.x, target.y)
            path = self.a_star(s, t, curr_loc, target)

            e = path[1]
            angle = sympy.atan2(e.y - curr_loc.y, e.x - curr_loc.x)
            distance = s.distance(e)
            return (distance, angle)


        minDist = float('inf')
        closestI = None
        for i in range(len(self.path)):
            if s.distance(self.path[i]) < minDist:
                minDist, closestI = s.distance(self.path[i]), i
        pt = self.path[closestI + 1]

        angle = sympy.atan2(pt.y - curr_loc.y, pt.x - curr_loc.x)
        distance = min(self.max_distance, s.distance(self.path[closestI + 1]))
        if distance > self.max_distance:
            distance = int(self.max_distance)

        distance = distance / 1.1

        return (distance, angle)
