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
        self.golf_map = sympy.Polygon(*json_obj["map"])
        sand_traps = list(json_obj["sand traps"])
        self.sand_traps = []
        for s in sand_traps:
            trap = sympy.Polygon(*s)
            self.sand_traps.append(trap)
        for s1 in self.sand_traps:
            assert self.golf_map.encloses(s1), "Golf map must enclose all sand traps"
            assert not s1.encloses(self.start), "Sand traps may not contain start"
            assert not s1.encloses(self.target), "Sand traps may not contain target"
            for s2 in self.sand_traps:
                if s1 == s2:
                    continue
                intersection = s1.intersection(s2)
                assert not intersection, "sand traps may not intersect"

        assert self.golf_map.encloses(self.start), "Start point doesn't lie inside map polygon"
        assert self.golf_map.encloses(self.target), "Target point doesn't lie inside map polygon"