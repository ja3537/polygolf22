import matplotlib.pyplot as plt
import json
import sys

if len(sys.argv) != 2:
    print("Usage: python plot_trials.py <log_file.json>")
    exit()

with open(sys.argv[1]) as f:
    log = json.loads(f.read())

with open(log["map"]) as f:
    map = json.loads(f.read())

plt.figure(facecolor="#bbddff")
plt.axis("off")
ax = plt.gca()
ax.invert_yaxis()

plt.fill(*list(zip(*map["map"])), facecolor="#bbff66", edgecolor="black", linewidth=1)

start_x, start_y = map["start"]
plt.plot(start_x, start_y, "bo")
target_x, target_y = map["target"]
plt.plot(target_x, target_y, "ro")

if "sand traps" in map:
    for trap in map["sand traps"]:
        plt.fill(*list(zip(*trap)), facecolor="#ffffcc", edgecolor="black", linewidth=1)


lands = log["landing_history"]["0"]
ends = log["ending_history"]["0"]

last_x = start_x
last_y = start_y
for n in range(len(ends)):
    land_x, land_y = lands[n]
    end_x, end_y = ends[n]
    plt.plot([last_x, land_x], [last_y, land_y], color="black", alpha=0.2)
    # plt.plot([land_x, end_x], [land_y, end_y], color="black", alpha=0.2)
    plt.arrow(land_x, land_y, end_x - land_x, end_y - land_y, color="black", alpha=0.5)
    last_x = end_x
    last_y = end_y


plt.savefig("render.png", dpi=200)
