import subprocess
import argparse
import json
from itertools import product
from glob import glob
from random import randint
import multiprocessing
import sys
import os.path as path
import os

LOG_DIR = "trials"

parser = argparse.ArgumentParser()
parser.add_argument("--skill", type=int, nargs="+", default=[10, 50, 100])
parser.add_argument("--trials", type=int, default=5)
parser.add_argument(
    "--maps",
    type=str,
    default="maps/**/*.json",
    help="glob pattern for matching map files",
)
args = parser.parse_args()


def run_trial(pargs):
    map, skill = pargs
    trials = []
    for trial in range(args.trials):
        proc = subprocess.run(
            f"python main.py --map {map} --skill {skill} --players 2 --seed {randint(0, sys.maxsize)} --disable_logging --no_gui",
            shell=True,
            stdout=subprocess.PIPE,
        )
        trials.append(json.loads(proc.stdout))
    print("Done:", map, skill)
    return trials


to_run = list(product(glob(args.maps), args.skill))
print(to_run)

# Run in parallel using all CPU cores
pool = multiprocessing.Pool(multiprocessing.cpu_count())
results = pool.map(run_trial, to_run)

to_render = {}
for result in results:
    logs = result
    map = logs[0]["map"]
    skill = logs[0]["skills"][0]
    if not map in to_render:
        to_render[map] = []
    to_render[map].append({"skill": skill, "logs": logs})

for map in to_render:
    target_path = path.join(LOG_DIR, map)
    os.makedirs(path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        f.write(json.dumps(to_render[map]))
