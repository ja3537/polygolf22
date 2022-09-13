from p5 import *
import json
import sympy
import numpy as np
import argparse
import os
import constants

args = None
FILE = None

map = []
golf_start = None
golf_target = None
mode = "green"
current_trap = None
traps = []

f = None  # STEP 1 Declare PFont variable


def draw_polygon(poly):
    begin_shape()
    for v in poly.vertices:
        vertex(float(v.x), float(v.y))
    # To add last side of polygon (last vertex to first vertex)
    vertex(poly.vertices[0].x, poly.vertices[0].y)
    end_shape()


def draw_point(p):
    point(float(p.x), float(p.y))


def draw_line(l):
    line(
        (float(l.points[0].x), float(l.points[0].y)),
        (float(l.points[1].x), float(l.points[1].y)),
    )


def draw_circle(c):
    circle(float(c.center.x), float(c.center.y), float(c.radius))


def setup():
    global f, args
    size(args.width, args.height)
    f = create_font("Arial.ttf", 16)  # STEP 2 Create Font
    text_font(f, 16)
    text_align("CENTER")


def draw():
    global map, golf_start, golf_target, f, args
    background(102)
    fill(0)
    text(
        "Press e to save and exit, s for start (red), t for target (green), a to add trap, d to delete trap",
        (int(width / 2), int(args.height * 0.9)),
    )
    text(
        "start clicking anywhere to draw map polygon",
        (int(width / 2), int(args.height * 0.95)),
    )

    stroke(0)
    fill(255)

    if map:
        poly = sympy.geometry.Polygon(*map)
        if len(map) == 1:
            draw_point(poly)
        elif len(map) == 2:
            draw_line(poly)
        elif len(map) > 2:

            draw_polygon(poly)
    if traps:
        fill(255, 255, 204)
        for trap in traps:
            poly = sympy.geometry.Polygon(*trap)
            if len(trap) == 1:
                draw_point(poly)
            elif len(trap) == 2:
                draw_line(poly)
            else:
                draw_polygon(poly)
        fill(255)
    if golf_start:
        sc = sympy.geometry.Circle(golf_start, 10)
        fill(255, 0, 0)
        draw_circle(sc)
        fill(255)
    if golf_target:
        ec = sympy.geometry.Circle(golf_target, 10)
        fill(0, 255, 0)
        draw_circle(ec)
        fill(255)

    text(f"Mode: {mode}", 100, 0)


def save():
    global map, golf_start, golf_target, traps, FILE
    map_list = []
    for p in map:
        tup = (float(p.x), float(p.y))
        map_list.append(tup)

    trap_list = []
    for trap in traps:
        trap_points = []
        trap_list.append(trap_points)
        for p in trap:
            trap_points.append((float(p.x), float(p.y)))

    save_dict = dict()
    save_dict["map"] = map_list
    save_dict["start"] = (float(golf_start.x), float(golf_start.y))
    save_dict["target"] = (float(golf_target.x), float(golf_target.y))
    save_dict["sand traps"] = trap_list

    with open(FILE, "w") as f:
        json.dump(save_dict, f)
    print("Auto-saved file {}".format(FILE))


def mouse_pressed():
    global map, golf_start, golf_target, mode, traps, current_trap
    if mode == "green":
        p = sympy.geometry.Point2D(mouse_x, mouse_y)
        print(mouse_x, mouse_y)
        map.append(p)
        print("New Map", map)
    else:
        if current_trap is None:
            current_trap = len(traps)
            traps.append([])
        p = sympy.geometry.Point2D(mouse_x, mouse_y)
        traps[current_trap].append(p)
        print(f"{len(traps)} traps, currently editing trap {current_trap}")
        print(traps[current_trap])
    save()


def key_pressed():
    global map, golf_start, golf_target, mode, traps, current_trap
    p = sympy.geometry.Point2D(mouse_x, mouse_y)
    # print(mouse_x, mouse_y)
    if key == "e":
        save()
        exit()
    elif key == "s":
        golf_start = p
        print("Start assigned")
        save()
    elif key == "t":
        golf_target = p
        print("Target assigned")
        save()
    elif key == "a":
        if mode == "green":
            mode = "traps"
            print("Switched to editing traps")
        else:
            # start new trap
            current_trap = None
    elif key == "g":
        if mode == "traps":
            mode = "green"
            current_trap = None
        else:
            map = []
        print("Switched to editing green")
    elif key == "d":
        # delete
        to_remove = []
        for trap in traps:
            poly = sympy.Polygon(*trap)
            if poly.encloses(p):
                to_remove.append(trap)
        for remove in to_remove:
            traps.remove(remove)
        current_trap = None
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", "-f", default="maps/temp.json", help="Path to export generated map"
    )
    parser.add_argument("--width", help="Width", type=int, default=constants.vis_width)
    parser.add_argument(
        "--height", help="Height", type=int, default=constants.vis_height
    )

    args = parser.parse_args()
    FILE = args.file

    golf_start = sympy.geometry.Point2D(args.width * 0.2, args.height * 0.2)
    golf_target = sympy.geometry.Point2D(args.width * 0.8, args.height * 0.8)

    dir = os.path.dirname(FILE)
    if dir:
        os.makedirs(dir, exist_ok=True)

    try:
        builtins.title = "Map Generation auto-saving to {}".format(FILE)
    except:
        pass
    run(frame_rate=60)
