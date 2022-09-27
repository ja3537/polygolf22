import matplotlib.pyplot as plt
import json
import sys
import os.path as path
import os
from itertools import pairwise, islice

PLOT_DIR = "plots"


def plot_trials(logs):
    num_skill_levels = len(logs)
    map_path = logs[0]["logs"][0]["map"]

    with open(map_path) as f:
        map = json.loads(f.read())

    xs = [x for x, _ in map["map"]]
    ys = [y for y, _ in map["map"]]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    fig = plt.figure(layout="constrained")
    fig.suptitle(f"Map: {map_path}")
    skill_subfigures = fig.subfigures(1, num_skill_levels)
    if num_skill_levels == 1:
        skill_subfigures = [skill_subfigures]

    for i, skill_log_obj in enumerate(logs):
        skill = skill_log_obj["skill"]
        skill_logs = skill_log_obj["logs"]

        subfig = skill_subfigures[i]
        subfig.suptitle(f"skill: {skill}")
        # ax1, ax2 = subfig.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})

        gs = subfig.add_gridspec(2,2, height_ratios=[2, 1])
        ax1 = subfig.add_subplot(gs[0, :])
        ax2 = subfig.add_subplot(gs[1, 0])
        ax3 = subfig.add_subplot(gs[1, 1])

        ax1.set_aspect(width / height)
        ax1.axis("off")
        ax1.invert_yaxis()
        ax1.set_facecolor("#bbddff")

        ax1.fill(
            *list(zip(*map["map"])), facecolor="#bbff66", edgecolor="black", linewidth=1
        )

        start_x, start_y = map["start"]
        ax1.plot(start_x, start_y, "bo")
        target_x, target_y = map["target"]
        ax1.plot(target_x, target_y, "ro")

        if "sand traps" in map:
            for trap in map["sand traps"]:
                ax1.fill(
                    *list(zip(*trap)),
                    facecolor="#ffffcc",
                    edgecolor="black",
                    linewidth=1,
                )

        opacity = max(1 / len(skill_logs), 0.05)
        for log in skill_logs:
            lands = log["landing_history"]["0"]
            ends = log["ending_history"]["0"]

            last_x = start_x
            last_y = start_y
            for n in range(len(ends)):
                land_x, land_y = lands[n]
                end_x, end_y = ends[n]
                ax1.plot(
                    [last_x, land_x], [last_y, land_y], color="blue", alpha=opacity, linewidth=1
                )
                ax1.arrow(
                    land_x,
                    land_y,
                    end_x - land_x,
                    end_y - land_y,
                    color="red",
                    alpha=0.5,
                    linewidth=1,
                    head_width=4,
                    head_length=4,
                    length_includes_head=True,
                )
                last_x = end_x
                last_y = end_y

        shot_count = {}
        for log in skill_logs:
            score = log["scores"][0]
            if not score in shot_count:
                shot_count[score] = 0
            shot_count[score] += 1

        ax2.bar(*list(zip(*shot_count.items())))
        ax2.set_xlabel("shots")
        ax2.set_ylabel("trials")
        ax2.set_xticks(list(shot_count.keys()))

        miss_count = {}
        for log in skill_logs:
            misses = sum(end_a == end_b for (end_a, end_b) in pairwise(log["ending_history"]["0"]))
            if not misses in miss_count:
                miss_count[misses] = 0
            miss_count[misses] += 1

        ax3.bar(*list(zip(*miss_count.items())))
        ax3.set_xlabel("misses")
        ax3.set_ylabel("trials")
        ax3.set_xticks(list(miss_count.keys()))
    
    map_basename = path.splitext(map_path)[0]
    target_path = path.join("plots", f"{map_basename}.svg")
    os.makedirs(path.dirname(target_path), exist_ok=True)
    plt.savefig(target_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_trials.py <log_file.json>")
        exit()

    with open(sys.argv[1]) as f:
        logs = json.loads(f.read())

    plot_trials(logs)
