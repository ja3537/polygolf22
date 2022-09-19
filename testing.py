import os
import sys
import subprocess
import json

maps = []
skills = [30, 60, 90]
groups = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"]
for group in groups:
    for (dirpath, dirnames, filenames) in os.walk(os.path.join("./maps", group)):
        for file in filenames:
            if file.endswith(".json"):
                maps.append(os.path.join(dirpath, file))
        break


def main():
    # subprocess.call(["mv",  "./players/default_player.py", "./players/tmp.py"])
    # subprocess.call(["cp",  "./2021_players/g2_player.py", "./players/default_player.py"])

    for map in maps:
        print("Map: " + map)
        for skill in skills:
            print("Skill: " + str(skill))
            res = subprocess.check_output(['python3', './main.py', '-ng', '-nb', '--skill', str(skill),  '-p', "2", '--map', map]).decode('utf-8').replace("'", '"').replace("(", '').replace(")", '')
            jsonResults = json.loads(res)
            if (jsonResults['player_states'][0] == "S"):
                if (len(jsonResults["player_states"]) > 1):
                    if (jsonResults['scores'][0] > jsonResults['scores'][1]):
                        printGreen("Score: " + str(jsonResults['scores'][0]) + " " + str(jsonResults['scores'][1]))
                    else:
                        printYellow("Score: " + str(jsonResults['scores'][0]) + " " + str(jsonResults['scores'][1]))
                        print("Time: " + str(jsonResults['total_time_sorted'][1]) + " " + str(jsonResults['total_time_sorted'][3]))

                else:
                    printGreen("Score: " + str(jsonResults['scores'][0]))
                    print("Time: " + str(jsonResults['total_time_sorted'][1]))
            else:
                printRed(res)
            print("")

    # subprocess.call(["mv",  "./players/tmp.py", "./players/default_player.py"])

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printRed(string):
    print(f"{bcolors.FAIL}{string}{bcolors.ENDC}")
def printGreen(string):
    print(f"{bcolors.OKGREEN}{string}{bcolors.ENDC}")
def printYellow(string):
    print(f"{bcolors.WARNING}{string}{bcolors.ENDC}")

if __name__ == '__main__':
    main()
