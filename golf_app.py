import logging
import os
import numpy as np
import sympy
import constants

from remi import App, gui


class GolfApp(App):
    def __init__(self, *args):
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(GolfApp, self).__init__(*args, static_file_path={'res': res_path})

    def convert_coord(self, coord):
        scale = min(self.scale.x*self.vis_width, self.scale.y*self.vis_height)
        coord = coord.translate(self.translate.x, self.translate.y)
        coord = coord.scale(x=scale, y=scale)
        coord = coord.translate((self.vis_width*self.padding_factor)/2, (self.vis_height*self.padding_factor)/2)
        return coord

    def draw_polygon(self, poly):
        svg_poly = gui.SvgPolygon(len(poly.vertices))
        for p in poly.vertices:
            p = self.convert_coord(p)
            svg_poly.add_coord(float(p.x), float(p.y))
        return svg_poly

    def draw_point(self, point):
        point = self.convert_coord(point)
        return gui.SvgCircle(float(point.x), float(point.y), 1.0)

    def draw_line(self, line):
        point1 = self.convert_coord(line.points[0])
        point2 = self.convert_coord(line.points[1])
        return gui.SvgLine(float(point1.x), float(point1.y), float(point2.x), float(point2.y))

    def draw_circle(self, circle):
        scale = min(self.scale.x*self.vis_width, self.scale.y*self.vis_height)
        center = self.convert_coord(circle.center)
        radius = scale*circle.radius
        return gui.SvgCircle(float(center.x), float(center.y), float(radius))

    def draw_text(self, point, text):
        point = self.convert_coord(point)
        return gui.SvgText(float(point.x), float(point.y), text)

    def main(self, *userdata):
        self.golf_game, start_automatic, self.logger = userdata
        self.golf_game.set_app(self)
        self.vis_width = constants.vis_width
        self.vis_height = constants.vis_height
        self.padding_factor = 1.+2*constants.vis_padding

        mainContainer = gui.Container(style={'width': '100%', 'height': '100%', 'overflow': 'auto', 'text-align': 'center'})
        mainContainer.style['justify-content'] = 'center'
        mainContainer.style['align-items'] = 'center'

        bt_hbox = gui.HBox()
        bt_hbox.attributes["class"] = "tophbox"
        play_step_bt = gui.Button("Play Step")
        play_turn_bt = gui.Button("Play Turn")
        play_all_bt = gui.Button("Play All")

        self.automatic_play = gui.CheckBoxLabel("Play Automatically", checked=start_automatic)
        self.automatic_play.attributes["class"] = "checkbox"

        self.view_drop_down = gui.DropDown(style={'padding': '5px', 'text-align': 'center', })
        for player_idx, player_name in enumerate(self.golf_game.player_names):
            self.view_drop_down.append(player_name, player_idx)
        self.view_drop_down.append("Empty Map", len(self.golf_game.player_names))

        self.view_drop_down.select_by_value(self.golf_game.get_current_player())
        self.view_drop_down.onchange.do(self.view_drop_down_changed)

        bt_hbox.append([play_step_bt, play_turn_bt, play_all_bt, self.automatic_play, self.view_drop_down])

        play_step_bt.onclick.do(self.play_step_bt_press)
        play_turn_bt.onclick.do(self.play_turn_bt_press)
        play_all_bt.onclick.do(self.play_all_bt_press)
        mainContainer.append(bt_hbox)
        self.labels = []
        self.labels.append(gui.Label("Polygolf: Ready to start", style={'margin': '5px auto'}))
        self.labels.append(gui.Label("", style={'margin': '5px auto'}))
        if self.golf_game.next_player is not None:
            self.golf_game.logger.debug("First turn {}".format(self.golf_game.get_current_player()))
            self.set_label_text("{}, First turn {}".format(self.get_label_text(), self.golf_game.get_current_player()))
        for label in self.labels:
            mainContainer.append(label)

        self.score_table = gui.TableWidget(2, len(self.golf_game.players), style={'margin': '5px auto'})
        self.update_score_table()

        for player_idx, _ in enumerate(self.golf_game.scores):
            self.score_table.item_at(0, player_idx).set_style("padding:0 10px")
            self.score_table.item_at(1, player_idx).set_style("padding:0 10px")
        mainContainer.append(self.score_table)

        self.load_map()
        self.svgplot = gui.Svg(width="{}vw".format(100*constants.vis_width_ratio*self.padding_factor), height="{}vh".format(100*constants.vis_height_ratio*self.padding_factor), style={'background-color': '#BBDDFF', 'margin': '0 auto', 'min-width': str(self.vis_width*self.padding_factor), 'min-height': str(self.vis_height*self.padding_factor)})
        self.svgplot.set_viewbox(0, 0, self.vis_width*self.padding_factor, self.vis_height*self.padding_factor)
        self.svgplot.attr_preserveAspectRatio = "xMidYMid"

        self.current_player_displayed = self.golf_game.get_current_player_idx()
        self.display_player(self.current_player_displayed)

        mainContainer.append(self.svgplot)
        return mainContainer

    def idle(self):
        if not self.golf_game.is_game_ended():
            if self.automatic_play.get_value():
                self.golf_game.play(run_stepwise=True)
        else:
            self.automatic_play.set_value(False)

    def load_map(self):
        self.golf_map = self.golf_game.golf.golf_map
        self.golf_start = self.golf_game.golf.start
        self.golf_target = self.golf_game.golf.target
        self.sand_traps = self.golf_game.golf.sand_traps

        bounds = self.golf_map.bounds
        xmin, ymin, xmax, ymax = list(map(float, bounds))
        translate_x = -(xmin+xmax)/2
        translate_y = -(ymin+ymax)/2
        self.translate = sympy.geometry.Point2D(translate_x, translate_y)
        self.scale = sympy.geometry.Point2D(1/(xmax-xmin), 1/(ymax-ymin))

        self.logger.info("Translating visualization by x={}, y={}".format(float(self.translate.x), float(self.translate.y)))
        self.logger.info("Base Scaling visualization by factors {}".format(float(self.scale.x), float(self.scale.y)))

    def reset_svgplot(self, full_refresh=False):
        if full_refresh or len(self.svgplot.children) == 0:
            self.svgplot.empty()
            p = self.draw_polygon(self.golf_map)
            # p.set_stroke(1, "black")
            p.set_fill("#BBFF66")
            self.svgplot.append(p)

            for s in self.sand_traps:
                trap = self.draw_polygon(s)
                trap.set_fill("#FFFFCC")
                self.svgplot.append(trap)

            unit_t = sympy.geometry.Triangle(asa=(60, 10, 60))
            golf_start_triangle = sympy.geometry.Triangle(*list(map(lambda p: p.translate(self.golf_start.x-unit_t.circumcenter.x, self.golf_start.y-unit_t.circumcenter.y), unit_t.vertices)))
            t = self.draw_polygon(golf_start_triangle)
            t.set_fill("red")
            self.svgplot.append(t)

            golf_target_circle = sympy.geometry.Circle(self.golf_target, constants.target_radius)
            c = self.draw_circle(golf_target_circle)
            c.set_fill("rgba(0,0,0,0.3)")
            self.svgplot.append(c)

            ps = self.draw_point(self.golf_start)
            self.svgplot.append(ps)

            pe = self.draw_point(self.golf_target)
            self.svgplot.append(pe)
            self.base_keys = list(self.svgplot.children.keys())
        else:
            for k in list(self.svgplot.children.keys()):
                if k not in self.base_keys:
                    self.svgplot.remove_child(self.svgplot.children[k])



    def display_player(self, player_idx, full_refresh=False):
        self.reset_svgplot(full_refresh)
        if player_idx is not None:
            for idx, step_play_dict in enumerate(self.golf_game.played[player_idx]):
                self.plot(step_play_dict, idx+1)

            self.set_label_text("Displaying {}".format(self.golf_game.player_names[player_idx]), 1)
            self.view_drop_down.select_by_key(player_idx)
        else:
            self.set_label_text("Displaying Empty Map".format(), 1)
            self.view_drop_down.select_by_key(len(self.golf_game.player_names))

        self.current_player_displayed = player_idx

    def view_drop_down_changed(self, widget, value):
        if self.automatic_play.get_value():
            self.automatic_play.set_value(False)
        player_idx = widget.get_key()
        if player_idx >= len(self.golf_game.player_names):
            player_idx = None
        self.display_player(player_idx)

    def match_display_with_game(self, full_refresh=False):
        player_idx = self.golf_game.get_current_player_idx()
        if player_idx is not None and self.current_player_displayed != player_idx:
            self.display_player(player_idx, full_refresh)

    def play_step_bt_press(self, widget):
        self.golf_game.play(run_stepwise=True)

    def play_turn_bt_press(self, widget):
        self.golf_game.play(run_stepwise=False)

    def play_all_bt_press(self, widget):
        self.golf_game.play_all()

    def plot(self, step_play_dict, idx):
        base_stroke_color = "0,0,0"
        if not step_play_dict["admissible"]:
            base_stroke_color = "255,0,0"
        stroke_color_air = "rgba({}, 0.2)".format(base_stroke_color)
        stroke_color_land = "rgba({}, 1)".format(base_stroke_color)

        if isinstance(step_play_dict["segment_air"], sympy.geometry.Segment2D):
            sa = self.draw_line(step_play_dict["segment_air"])
            sa.set_stroke(3, stroke_color_air)
            self.svgplot.append(sa)
        elif isinstance(step_play_dict["segment_air"], sympy.geometry.Point2D):
            sa = self.draw_point(step_play_dict["segment_air"])
            sa.set_stroke(3, stroke_color_air)
            self.svgplot.append(sa)

        if isinstance(step_play_dict["segment_land"], sympy.geometry.Segment2D):
            sl = self.draw_line(step_play_dict["segment_land"])
            sl.set_stroke(3, stroke_color_land)
            self.svgplot.append(sl)
            text = self.draw_text(step_play_dict["segment_land"].midpoint, str(idx))
            text.set_stroke(1, stroke_color_land)
            self.svgplot.append(text)
        elif isinstance(step_play_dict["segment_land"], sympy.geometry.Point2D):
            sl = self.draw_point(step_play_dict["segment_land"])
            sl.set_stroke(3, stroke_color_land)
            self.svgplot.append(sl)
            text = self.draw_text(step_play_dict["segment_land"], str(idx))
            text.set_stroke(1, stroke_color_land)
            self.svgplot.append(text)

    def update_score_table(self):
        for player_idx, score in enumerate(self.golf_game.scores):
            self.score_table.item_at(0, player_idx).set_text("{}, {}".format(self.golf_game.player_names[player_idx], self.golf_game.skills[player_idx]))
            self.score_table.item_at(1, player_idx).set_text("{}, {}".format(score, self.golf_game.player_states[player_idx]))

    def set_label_text(self, text, label_num=0):
        self.labels[label_num].set_text(text)

    def get_label_text(self, label_num=0):
        return self.labels[label_num].get_text()
