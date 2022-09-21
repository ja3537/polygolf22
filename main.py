import logging
import os
import argparse
import constants
from golf_game import GolfGame
import json
from numpyencoder import NumpyEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map", "-m", default=constants.default_map, help="Path to map json file"
    )
    parser.add_argument(
        "--skill",
        type=int,
        help="Skill to use, don't specify to randomly choose between min and max",
    )
    parser.add_argument(
        "--automatic",
        action="store_true",
        help="Start playing automatically in GUI mode",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=2,
        help="Seed used by random number generator, specify 0 to use no seed and have different random behavior on each launch",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to start, specify -1 to auto-assign",
    )
    parser.add_argument(
        "--address", "-a", type=str, default="127.0.0.1", help="Address"
    )
    parser.add_argument(
        "--no_browser",
        "-nb",
        action="store_true",
        help="Disable browser launching in GUI mode",
    )
    parser.add_argument("--no_gui", "-ng", action="store_true", help="Disable GUI")
    parser.add_argument(
        "--log_path",
        default="log",
        help="Directory path to dump log files, filepath if disable_logging is false",
    )
    parser.add_argument(
        "--disable_timeout",
        "-time",
        action="store_true",
        help="Disable Timeout in non GUI mode",
    )
    parser.add_argument(
        "--disable_logging",
        action="store_true",
        help="Disable Logging, log_path becomes path to file",
    )
    parser.add_argument(
        "--players",
        "-p",
        default=["d"],
        nargs="+",
        help="List of players space separated",
    )
    args = parser.parse_args()
    player_list = tuple(args.players)
    del args.players

    if args.disable_logging:
        if args.log_path == "log":
            args.log_path = "results.log"

    golf_game = GolfGame(player_list, args)
    if not golf_game.use_gui:
        golf_game.play_all()
        result = golf_game.get_state()
        print(json.dumps(result, cls=NumpyEncoder))
