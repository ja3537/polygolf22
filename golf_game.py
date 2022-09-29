import logging
import os
import time
import signal
import numpy as np
import sympy
from remi import start
from golf_app import GolfApp
from golf_map import GolfMap
import constants
from utils import *
from players.default_player import Player as DefaultPLayer
from players.g1_player import Player as G1_Player
from players.g2_player import Player as G2_Player
from players.g3_player import Player as G3_Player
from players.g4_player import Player as G4_Player
from players.g5_player import Player as G5_Player
from players.old_g2_player import Player as G6_Player
from players.g7_player import Player as G7_Player
from players.g8_player import Player as G8_Player
from players.g9_player import Player as G9_Player
from players.g5_2nd_player import Player as G10_Player


return_vals = [
    "player_names",
    "map",
    "skills",
    "scores",
    "player_states",
    "distances_from_target",
    "distance_source_to_target",
    "start",
    "target",
    "penalties",
    "timeout_count",
    "error_count",
    "winner_list",
    "total_time_sorted",
    "landing_history",
    "ending_history",
]


class GolfGame:
    def __init__(self, player_list, args):
        self.use_gui = not (args.no_gui)
        self.do_logging = not (args.disable_logging)
        if not self.use_gui:
            self.use_timeout = not (args.disable_timeout)
        else:
            self.use_timeout = False

        self.logger = logging.getLogger(__name__)
        # create file handler which logs even debug messages
        if self.do_logging:
            self.logger.setLevel(logging.DEBUG)
            self.log_dir = args.log_path
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(self.log_dir, "debug.log"), mode="w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(message)s"))
            fh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(fh)
            result_path = os.path.join(self.log_dir, "results.log")
            rfh = logging.FileHandler(result_path, mode="w")
            rfh.setLevel(logging.INFO)
            rfh.setFormatter(logging.Formatter("%(message)s"))
            rfh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(rfh)
        else:
            if args.log_path:
                self.logger.setLevel(logging.INFO)
                result_path = args.log_path
                self.log_dir = os.path.dirname(result_path)
                if self.log_dir:
                    os.makedirs(self.log_dir, exist_ok=True)
                rfh = logging.FileHandler(result_path, mode="w")
                rfh.setLevel(logging.INFO)
                rfh.setFormatter(logging.Formatter("%(message)s"))
                rfh.addFilter(MainLoggingFilter(__name__))
                self.logger.addHandler(rfh)
            else:
                self.logger.setLevel(logging.ERROR)
                self.logger.disabled = True

        if args.seed == 0:
            args.seed = None
            self.logger.info("Initialise random number generator with no seed")
        else:
            self.logger.info(
                "Initialise random number generator with seed {}".format(args.seed)
            )

        self.rng = np.random.default_rng(args.seed)

        self.golf = GolfMap(args.map, self.logger)
        self.players = []
        self.player_names = []
        self.skills = []
        self.player_states = []
        self.played = []
        self.curr_locs = []
        self.scores = []
        self.penalties = []
        self.next_player = None

        self.landing_history = {player: [] for player in range(len(player_list))}
        self.ending_history = {player: [] for player in range(len(player_list))}

        self.time_taken = []
        self.timeout_count = []
        self.error_count = []

        self.winner_list = None
        self.total_time_sorted = None
        self.distances_from_target = None
        self.map = self.golf.map_filepath
        self.start = np.array(self.golf.start).astype(float).tolist()
        self.target = np.array(self.golf.target).astype(float).tolist()
        self.distance_source_to_target = float(
            self.golf.start.distance(self.golf.target)
        )

        self.processing_turn = False
        self.end_message_printed = False

        self.__add_players(player_list, args.skill)
        self.next_player = self.__assign_next_player()

        if self.use_gui:
            config = dict()
            config["address"] = args.address
            config["start_browser"] = not (args.no_browser)
            config["update_interval"] = 0.5
            config["userdata"] = (self, args.automatic, self.logger)
            if args.port != -1:
                config["port"] = args.port
            start(GolfApp, **config)
        else:
            self.logger.debug("No GUI flag specified")

    def set_app(self, golf_app):
        self.golf_app = golf_app

    def get_current_player(self):
        if self.next_player is not None:
            return self.player_names[self.next_player]
        return None

    def get_current_player_idx(self):
        return self.next_player

    def __add_players(self, player_list, skill=None):
        player_count = dict()
        for player_name in player_list:
            if player_name not in player_count:
                player_count[player_name] = 0
            player_count[player_name] += 1

        count_used = {k: 0 for k in player_count}
        for player_name in player_list:
            if player_name in constants.possible_players:
                if player_name.lower() == "d":
                    player_class = DefaultPLayer
                    base_player_name = "Default Player"
                else:
                    player_class = eval("G{}_Player".format(player_name))
                    base_player_name = "Group {}".format(player_name)
                count_used[player_name] += 1
                if player_count[player_name] == 1:
                    self.__add_player(
                        player_class,
                        "{}".format(base_player_name),
                        base_player_name=base_player_name,
                        skill=skill,
                    )
                else:
                    self.__add_player(
                        player_class,
                        "{}.{}".format(base_player_name, count_used[player_name]),
                        base_player_name=base_player_name,
                        skill=skill,
                    )
            else:
                self.logger.error(
                    "Failed to insert player {} since invalid player name provided.".format(
                        player_name
                    )
                )

    def __add_player(self, player_class, player_name, base_player_name, skill=None):
        if player_name not in self.player_names:
            if (
                skill is None
                or skill < constants.min_skill
                or skill > constants.max_skill
            ):
                skill = self.rng.integers(constants.min_skill, constants.max_skill + 1)
            self.logger.info(
                "Adding player {} from class {} with skill {}".format(
                    player_name, player_class.__module__, skill
                )
            )
            precomp_dir = os.path.join("precomp", base_player_name)
            os.makedirs(precomp_dir, exist_ok=True)
            player_map_path = slugify(self.golf.map_filepath)

            is_timeout = False
            if self.use_timeout:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(constants.timeout)
            try:
                start_time = time.time()
                player = player_class(
                    skill=skill,
                    rng=self.rng,
                    logger=self.__get_player_logger(player_name),
                    golf_map=self.golf.golf_map.copy(),
                    start=self.golf.start.copy(),
                    target=self.golf.target.copy(),
                    sand_traps=self.golf.sand_traps,
                    map_path=player_map_path,
                    precomp_dir=precomp_dir,
                )
                if self.use_timeout:
                    signal.alarm(0)  # Clear alarm
            except TimeoutException:
                is_timeout = True
                player = None
                self.logger.error(
                    "Initialization Timeout {} since {:.3f}s reached.".format(
                        player_name, constants.timeout
                    )
                )

            init_time = time.time() - start_time

            if not is_timeout:
                self.logger.info(
                    "Initializing player {} took {:.3f}s".format(player_name, init_time)
                )
            self.players.append(player)
            self.player_names.append(player_name)
            self.skills.append(skill)
            self.player_states.append("NP")
            self.played.append([])
            self.curr_locs.append(self.golf.start.copy())
            self.scores.append(0)
            self.penalties.append(0)
            self.time_taken.append([init_time])
            self.timeout_count.append(0)
            self.error_count.append(0)

            if is_timeout:
                player_idx = len(self.players) - 1
                self.timeout_count[player_idx] += 1
                self.player_states[player_idx] = "F"
                self.scores[player_idx] = constants.max_tries
        else:
            self.logger.error(
                "Failed to insert player as another player with name {} exists.".format(
                    player_name
                )
            )

    def __get_player_logger(self, player_name):
        player_logger = logging.getLogger("{}.{}".format(__name__, player_name))

        if self.do_logging:
            player_logger.setLevel(logging.INFO)
            # add handler to self.logger with filtering
            player_fh = logging.FileHandler(
                os.path.join(self.log_dir, "{}.log".format(player_name)), mode="w"
            )
            player_fh.setLevel(logging.DEBUG)
            player_fh.setFormatter(logging.Formatter("%(message)s"))
            player_fh.addFilter(PlayerLoggingFilter(player_name))
            self.logger.addHandler(player_fh)
        else:
            player_logger.setLevel(logging.ERROR)
            player_logger.disabled = True

        return player_logger

    def is_game_ended(self):
        return np.all([x in constants.end_player_states for x in self.player_states])

    def __game_end(self):
        if not self.end_message_printed and self.is_game_ended():
            self.end_message_printed = True
            self.logger.info("Game ended as each player finished playing")
            if self.use_gui:
                self.golf_app.set_label_text(
                    "Game ended as each player finished playing"
                )

            total_time = np.zeros(len(self.players))
            for player_idx, player_time_taken in enumerate(self.time_taken):
                player_time_taken_flatten = np.array(player_time_taken)

                if player_time_taken_flatten.size == 0:
                    player_time_taken_flatten = np.zeros(2)
                elif player_time_taken_flatten.size == 1:
                    player_time_taken_flatten = np.append(player_time_taken_flatten, 0)

                self.logger.info(
                    "{} total time {:.3f}s, init time {:.3f}s, total step time: {:.3f}s".format(
                        self.player_names[player_idx],
                        np.sum(player_time_taken_flatten),
                        player_time_taken_flatten[0],
                        np.sum(player_time_taken_flatten[1:]),
                    )
                )

                self.logger.info(
                    "{} took {} steps, avg time {:.3f}s, avg step time {:.3f}s, max step time {:.3f}s".format(
                        self.player_names[player_idx],
                        player_time_taken_flatten.size - 1,
                        np.mean(player_time_taken_flatten),
                        np.mean(player_time_taken_flatten[1:]),
                        np.amax(player_time_taken_flatten[1:]),
                    )
                )
                total_time[player_idx] = np.sum(player_time_taken_flatten)
            self.logger.info(
                "Total time taken by all players {:.3f}s".format(np.sum(total_time))
            )
            total_time_sort_idx = np.argsort(total_time)[::-1]
            self.total_time_sorted = [
                (self.player_names[player_idx], total_time[player_idx])
                for player_idx in total_time_sort_idx
            ]
            self.logger.info("Players sorted by total time")
            for (player_name, player_total_time) in self.total_time_sorted:
                self.logger.info(
                    "{} took {:.3f}s".format(player_name, player_total_time)
                )

            for player_idx, player_timeout_count in enumerate(self.timeout_count):
                if player_timeout_count > 0:
                    self.logger.info(
                        "{} timed out {} times".format(
                            self.player_names[player_idx], player_timeout_count
                        )
                    )

            for player_idx, player_error_count in enumerate(self.error_count):
                if player_error_count > 0:
                    self.logger.info(
                        "{} had exceptions {} times".format(
                            self.player_names[player_idx], player_error_count
                        )
                    )

            for player_idx, penalty in enumerate(self.penalties):
                if penalty != 0:
                    self.logger.info(
                        "{} had {} {}".format(
                            self.player_names[player_idx],
                            penalty,
                            "penalties" if penalty != 1 else "penalty",
                        )
                    )
                else:
                    self.logger.info(
                        "{} had no penalty".format(self.player_names[player_idx])
                    )

            for player_idx, score in enumerate(self.scores):
                self.logger.info(
                    "{} score: {}".format(self.player_names[player_idx], score)
                )

            for player_idx, player_state in enumerate(self.player_states):
                self.logger.info(
                    "{} final player state: {}".format(
                        self.player_names[player_idx], player_state
                    )
                )

            self.logger.info(
                "Distance from source to target {:.3f}".format(
                    self.distance_source_to_target
                )
            )
            self.distances_from_target = [
                float(self.curr_locs[player_idx].distance(self.golf.target))
                if self.player_states[player_idx] != "S"
                else 0.0
                for player_idx in range(len(self.player_names))
            ]
            for player_idx, distance_from_target in enumerate(
                self.distances_from_target
            ):
                self.logger.info(
                    "{} final distance from target: {:.3f}".format(
                        self.player_names[player_idx], distance_from_target
                    )
                )

            winner_list_idx = [
                idx
                for idx in range(len(self.player_names))
                if self.player_states[idx] == "S"
            ]  # winner(s) should solve the game
            if len(winner_list_idx):
                scores_array = np.array(
                    [self.scores[idx] for idx in winner_list_idx], dtype=np.int
                )
                modified_winner_list_idx = np.argwhere(
                    scores_array == np.amin(scores_array)
                ).squeeze(
                    axis=1
                )  # winner(s) should have min score too
                final_winner_list_idx = [
                    winner_list_idx[i] for i in modified_winner_list_idx
                ]
                self.winner_list = [self.player_names[i] for i in final_winner_list_idx]
            else:
                self.winner_list = []

            self.logger.info(
                "Winner{}: {}".format(
                    "s" if len(self.winner_list) > 1 else "",
                    ", ".join(self.winner_list),
                )
            )
            if self.use_gui:
                self.golf_app.set_label_text(
                    "Winner{}: {}".format(
                        "s" if len(self.winner_list) > 1 else "",
                        ", ".join(self.winner_list),
                    ),
                    label_num=1,
                )

    def __assign_next_player(self):
        # randomly select among valid players
        valid_players = [
            i
            for i, s in enumerate(self.player_states)
            if s not in constants.end_player_states
        ]
        if valid_players:
            # return valid_players[self.rng.integers(0, valid_players.size)]
            return valid_players[0]
        return None

    def __turn_end(self):
        self.processing_turn = False

        self.next_player = self.__assign_next_player()
        if self.next_player is not None:
            self.logger.debug(
                "Next turn {}".format(self.player_names[self.next_player])
            )
            if self.use_gui:
                self.golf_app.set_label_text(
                    "Next turn {}".format(self.player_names[self.next_player])
                )

    def get_state(self):
        return_dict = dict()
        for val in return_vals:
            value = getattr(self, val)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            return_dict[val] = value
        return return_dict

    def play_all(self):
        if not self.is_game_ended():
            self.logger.debug("Playing all turns")
            if self.use_gui:
                self.golf_app.set_label_text("Playing all turns")
                self.golf_app.set_label_text("Processing...", label_num=1)
                self.golf_app.do_gui_update()
            while not self.is_game_ended():
                self.play(run_stepwise=False, do_update=False, display_end=False)
            if self.use_gui:
                self.golf_app.display_player(len(self.player_names) - 1)
                self.golf_app.update_score_table()
            self.__game_end()
        elif not self.end_message_printed:
            self.__game_end()

    def play(self, run_stepwise=False, do_update=True, display_end=True):
        if not self.processing_turn:
            if not self.is_game_ended():
                if self.player_states[self.next_player] in constants.end_player_states:
                    self.logger.debug(
                        "Can't pass to the {}, as the player's game finished".format(
                            self.player_names[self.next_player]
                        )
                    )
                    self.next_player = self.__assign_next_player()
                    self.logger.debug(
                        "Assigned new player {}".format(
                            self.player_names[self.next_player]
                        )
                    )

                self.logger.debug(
                    "Current turn {}".format(self.player_names[self.next_player])
                )
                if self.use_gui:
                    self.golf_app.set_label_text(
                        "Current turn {}".format(self.player_names[self.next_player])
                    )

                self.processing_turn = True
                self.player_states[self.next_player] = "P"

            else:
                if display_end and not self.end_message_printed:
                    self.__game_end()
                return

        if run_stepwise:
            self.golf_app.match_display_with_game()
            pass_next = self.__step(self.next_player, do_update)
            if pass_next:
                self.__turn_end()

        else:
            if do_update and self.use_gui:
                self.golf_app.set_label_text("Processing...", label_num=1)
                self.golf_app.do_gui_update()
            pass_next = False
            while not pass_next:
                pass_next = self.__step(self.next_player, do_update=False)
            if do_update and self.use_gui:
                self.golf_app.display_player(self.next_player)
                self.golf_app.update_score_table()
            self.__turn_end()

        if display_end and self.is_game_ended() and not self.end_message_printed:
            self.__game_end()

    def __check_action(self, returned_action):
        if not returned_action:
            return False
        is_valid = False
        if isiterable(returned_action) and count_iterable(returned_action) == 2:
            if np.all(
                [x is not None and sympy.simplify(x).is_real for x in returned_action]
            ):
                is_valid = True

        return is_valid

    def __step(self, player_idx, do_update=True):
        pass_next = False
        if (
            self.player_states[player_idx] in ["F", "S"]
            or self.scores[player_idx] >= constants.max_tries
        ):
            pass_next = True
        else:
            self.scores[player_idx] += 1
            try:
                time_limit_already_exceeded = False
                if self.use_timeout:
                    remaining_time = np.ceil(
                        constants.timeout - np.sum(self.time_taken[player_idx])
                    ).astype(int)
                    if remaining_time > 0:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(remaining_time)
                    else:
                        time_limit_already_exceeded = True
                if not time_limit_already_exceeded:
                    try:
                        start_time = time.time()
                        prev_loc = None
                        prev_landing_point = None
                        prev_admissible = None
                        if len(self.played[player_idx]):
                            last_step_play_dict = self.played[player_idx][-1]
                            prev_loc = last_step_play_dict["last_location"].copy()
                            prev_landing_point = last_step_play_dict[
                                "observed_landing_point"
                            ].copy()
                            prev_admissible = last_step_play_dict["admissible"]
                        returned_action = self.players[player_idx].play(
                            score=self.scores[player_idx],
                            golf_map=self.golf.golf_map.copy(),
                            target=self.golf.target.copy(),
                            sand_traps=self.golf.sand_traps,
                            curr_loc=self.curr_locs[player_idx].copy(),
                            prev_loc=prev_loc,
                            prev_landing_point=prev_landing_point,
                            prev_admissible=prev_admissible,
                        )
                        if self.use_timeout:
                            signal.alarm(0)  # Clear alarm
                    except TimeoutException:
                        self.logger.error(
                            "Timeout {} since {:.3f}s reached.".format(
                                self.player_names[player_idx], constants.timeout
                            )
                        )
                        returned_action = None
                        self.timeout_count[player_idx] += 1
                        self.scores[player_idx] = constants.max_tries

                    step_time = time.time() - start_time
                    self.time_taken[player_idx].append(step_time)

                else:
                    self.logger.error(
                        "Skipping {} since time limit of {:.3f}s already exceeded and already ran for {:.3f}s.".format(
                            self.player_names[player_idx],
                            constants.timeout,
                            np.sum(self.time_taken[player_idx]),
                        )
                    )
                    returned_action = None
                    self.timeout_count[player_idx] += 1
                    self.scores[player_idx] = constants.max_tries
            except Exception as e:
                self.logger.error(e, exc_info=True)
                returned_action = None
                self.error_count[player_idx] += 1
                self.scores[player_idx] = constants.max_tries

            is_valid_action = self.__check_action(returned_action)
            if is_valid_action:
                distance, angle = returned_action
                self.logger.debug(
                    "Received Distance: {:.3f}, Angle: {:.3f} from {} in {:.3f}s".format(
                        float(distance),
                        float(angle),
                        self.player_names[player_idx],
                        step_time,
                    )
                )
                if self.use_gui:
                    self.golf_app.set_label_text(
                        "{}, ({:.2f},{:.2f})".format(
                            self.golf_app.get_label_text(),
                            float(distance),
                            float(angle),
                        )
                    )
                step_play_dict = self.__move(distance, angle, player_idx)

                self.curr_locs[player_idx] = step_play_dict["observed_final_point"]
                if not step_play_dict["admissible"]:
                    self.penalties[player_idx] += 1

                self.played[player_idx].append(step_play_dict)

                if do_update and self.use_gui:
                    self.golf_app.plot(step_play_dict, len(self.played[player_idx]))

                if step_play_dict["reached_target"]:
                    self.logger.info(
                        "{} reached Target with score {}".format(
                            self.player_names[player_idx], self.scores[player_idx]
                        )
                    )
                    if self.use_gui:
                        self.golf_app.set_label_text(
                            "{} reached Target with score {}".format(
                                self.player_names[player_idx], self.scores[player_idx]
                            )
                        )
                    self.player_states[player_idx] = "S"
                    pass_next = True
                elif self.scores[player_idx] >= constants.max_tries:
                    self.logger.info(
                        "{} failed since it used {} max tries".format(
                            self.player_names[player_idx], constants.max_tries
                        )
                    )
                    if self.use_gui:
                        self.golf_app.set_label_text(
                            "{} failed since it used {} max tries".format(
                                self.player_names[player_idx], constants.max_tries
                            )
                        )
                    self.player_states[player_idx] = "F"
                    pass_next = True
            else:
                self.logger.info(
                    "{} failed since provided invalid action {}".format(
                        self.player_names[player_idx], returned_action
                    )
                )
                if self.use_gui:
                    self.golf_app.set_label_text(
                        "{} failed since provided invalid action {}".format(
                            self.player_names[player_idx], returned_action
                        )
                    )
                self.player_states[player_idx] = "F"
                pass_next = True

        if do_update and self.use_gui:
            self.golf_app.update_score_table()

        return pass_next

    def __move(self, distance, angle, player_idx):
        curr_loc = self.curr_locs[player_idx]
        in_sand = False
        for s in self.golf.sand_traps:
            if s.encloses(curr_loc):
                in_sand = True
                break
        if not in_sand:
            actual_distance = self.rng.normal(
                distance, distance / self.skills[player_idx]
            )
            actual_angle = self.rng.normal(angle, 1 / (2 * self.skills[player_idx]))
        else:
            distance = min(
                distance, (constants.max_dist + self.skills[player_idx]) / 2
            )  # backwards compatibility exception for last year's players
            actual_distance = self.rng.normal(
                distance, 2 * distance / self.skills[player_idx]
            )
            actual_angle = self.rng.normal(angle, 2 / (2 * self.skills[player_idx]))

        if (
            distance <= constants.max_dist + self.skills[player_idx]
            and distance >= constants.min_putter_dist
        ):
            landing_point = sympy.Point2D(
                curr_loc.x + actual_distance * sympy.cos(actual_angle),
                curr_loc.y + actual_distance * sympy.sin(actual_angle),
            )
            final_point = sympy.Point2D(
                curr_loc.x
                + (1.0 + constants.extra_roll)
                * actual_distance
                * sympy.cos(actual_angle),
                curr_loc.y
                + (1.0 + constants.extra_roll)
                * actual_distance
                * sympy.sin(actual_angle),
            )

        elif distance < constants.min_putter_dist:
            if not in_sand:
                self.logger.debug(
                    "Using Putter as provided distance {:.3f} less than {}".format(
                        float(distance), constants.min_putter_dist
                    )
                )
                landing_point = curr_loc
                final_point = sympy.Point2D(
                    curr_loc.x + actual_distance * sympy.cos(actual_angle),
                    curr_loc.y + actual_distance * sympy.sin(actual_angle),
                )
            else:
                landing_point = sympy.Point2D(
                    curr_loc.x + actual_distance * sympy.cos(actual_angle),
                    curr_loc.y + actual_distance * sympy.sin(actual_angle),
                )
                final_point = sympy.Point2D(
                    curr_loc.x
                    + (1.0 + constants.extra_roll)
                    * actual_distance
                    * sympy.cos(actual_angle),
                    curr_loc.y
                    + (1.0 + constants.extra_roll)
                    * actual_distance
                    * sympy.sin(actual_angle),
                )

        else:
            self.logger.debug(
                "Provide invalid distance {:.3f}, distance should be < {}".format(
                    float(distance), constants.max_dist + self.skills[player_idx]
                )
            )
            landing_point = curr_loc
            final_point = curr_loc

        self.logger.debug(
            "Observed Distance: {:.3f}, Angle: {:.3f}".format(
                actual_distance, actual_angle
            )
        )

        land_in_sand = False
        for s in self.golf.sand_traps:  # checks for landing in sand
            if s.encloses(landing_point):
                land_in_sand = True
                final_point = landing_point
                break

        segment_land = sympy.geometry.Segment2D(landing_point, final_point)
        intersections = []
        roll_in_sand = False
        for s in self.golf.sand_traps:  # check for rolling into sand traps
            intersections.extend(s.intersection(segment_land))
        minp = None
        mind = float("inf")
        for p in intersections:
            if type(p) == sympy.geometry.Point2D:
                if p.distance(landing_point) < mind:
                    minp = p
                    mind = p.distance(landing_point)

        if minp is not None:  # rolled into sand trap
            roll_in_sand = True
            if minp.distance(final_point) > 0.01:
                sand_roll = sympy.Segment2D(minp, final_point)
                final_point = final_point = sympy.Point2D(
                    minp.x + 0.01 * sympy.cos(actual_angle),
                    minp.y + 0.01 * sympy.sin(actual_angle),
                )  # roll 1cm
                segment_land = sympy.geometry.Segment2D(landing_point, final_point)

        segment_air = sympy.geometry.Segment2D(curr_loc, landing_point)

        if segment_land.distance(self.golf.target) <= constants.target_radius:
            reached_target = True
            final_point = segment_land.projection(self.golf.target)
            segment_land = sympy.geometry.Segment2D(landing_point, final_point)
        else:
            reached_target = False

        admissible = False
        if self.golf.golf_map.encloses(
            segment_land
        ) and not self.golf.golf_map.intersection(segment_land):
            admissible = True
            observed_final_point = final_point
            observed_landing_point = landing_point
        else:
            observed_final_point = curr_loc
            observed_landing_point = curr_loc

        reached_target = reached_target and admissible

        step_play_dict = dict()
        step_play_dict["segment_air"] = segment_air
        step_play_dict["segment_land"] = segment_land
        step_play_dict["last_location"] = self.curr_locs[player_idx]
        step_play_dict["observed_final_point"] = observed_final_point
        step_play_dict["observed_landing_point"] = observed_landing_point
        step_play_dict["admissible"] = admissible
        step_play_dict["reached_target"] = reached_target
        step_play_dict["land_in_sand"] = land_in_sand
        step_play_dict["roll_in_sand"] = roll_in_sand


        self.landing_history[player_idx].append(
            (float(observed_landing_point.x), float(observed_landing_point.y))
        )
        self.ending_history[player_idx].append(
            (float(observed_final_point.x), float(observed_final_point.y))
        )

        return step_play_dict

