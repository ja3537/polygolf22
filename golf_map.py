import logging
import os
import numpy as np
import sympy
import json
import constants

class GolfMap:
    def __init__(self, map_filepath, logger) -> None:
        self.logger = logger

        self.logger.info("Map file loaded: {}".format(map_filepath))
        with open(map_filepath, "r") as f:
            json_obj = json.load(f)
        self.map_filepath = map_filepath
        self.start = sympy.geometry.Point2D(*json_obj["start"])
        self.target = sympy.geometry.Point2D(*json_obj["target"])
        print("map")
        print(*json_obj["map"])
        self.golf_map = sympy.Polygon(*json_obj["map"])
        sand_traps = list(json_obj["sand traps"])
        self.sand_traps = []
        print(sand_traps)
        print(sand_traps[0])
        for s in sand_traps:
            trap = sympy.Polygon(*s)
            self.sand_traps.append(trap)
        assert self.golf_map.encloses(self.start), "Start point doesn't lie inside map polygon"
        assert self.golf_map.encloses(self.target), "Target point doesn't lie inside map polygon"