import subprocess
import sys

# def install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#
# install("numpy")

import os.path

import asyncio
import random
import numpy as np

import random
import math

from hashlib import sha1

from lia.enums import *
from lia.api import *
from lia import constants
from lia import math_util
from lia.bot import Bot
from lia.networking_client import connect

from collections import defaultdict

global VOJAK_q_table
global NABIRALEC_q_table

botWasCloseToFood = defaultdict(lambda: False)
global lastFood
global lastPobranih
global pobranih
global a

lastFood = 0
lastPobranih = 0
pobranih = 0
a = True

def FindMaxInArray(arr):
    vse = []

    last = 0

    for location, value in np.ndenumerate(arr):

        vse.append((location, value))

    print("A", "done")

    arrSorted = sorted(vse, key=lambda x: x[1])

    for i in arrSorted[:100]:
        print("A", i)

    print("A", "|||||||||||||||")

    for i in arrSorted[-100:]:
        print("A", i)

def SaveIfBetter(arr, name, newBestScore):
    if os.path.isfile(name + '_qTable_BEST.npy'):
        with open(name + '_best_score.txt', 'r') as file:
            lastBestScore = file.readline()

        print("A", lastBestScore, newBestScore)

        if int(lastBestScore) < int(newBestScore):
            with open(name + '_best_score.txt', 'w') as file:
                file.write(str(newBestScore))
            np.save(name + "_qTable_BEST.npy", arr, allow_pickle=True)

            print("A", name + " new best saved")
    else:
        with open(name + '_best_score.txt', 'w') as file:
            file.write(str(newBestScore))
        np.save(name + "_qTable_BEST.npy", arr, allow_pickle=True)



def GetDistanceToWall(api, unit):
    radians = math.radians(unit["orientationAngle"])

    moveY = math.sin(radians)/10
    moveX = math.cos(radians)/10

    x = unit["x"]
    y = unit["y"]

    while not constants.MAP[int(x)][int(y)]:
        x += moveX
        y += moveY

        if int(x) > constants.MAP_WIDTH-1 or int(y) > constants.MAP_HEIGHT-1 or int(x) < 0 or int(y) < 0:
            break

    distanceToObsticle = math_util.distance(unit["x"], unit["y"], x, y)


    #if unit["id"] == 1:
    #if distanceToObsticle < 2:
    #api.say_something(unit["id"], str(round(distanceToObsticle, 2)) + "|" + str(round(unit["x"], 2)) + "|" + str(round(unit["y"], 2)) + "|" + str(radians))
    #print("A", unit["orientationAngle"], distanceToObsticle)

    return distanceToObsticle


def Get_discrete_state_Vojak(state):
    discrete_state = (state - VOJAK_OBSERVATION_SPACE_LOW)/VOJAK_discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

def GetAction_Vojak(discrete_state):
    global VOJAK_q_table
    return np.argmax(VOJAK_q_table[discrete_state])


def Get_discrete_state_Nabiralec(state):
    discrete_state = (state - NABIRALEC_OBSERVATION_SPACE_LOW)/NABIRALEC_discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

def GetAction_Nabiralec(discrete_state):
    global NABIRALEC_q_table
    #print("A", NABIRALEC_q_table[discrete_state])
    return np.argmax(NABIRALEC_q_table[discrete_state])


def GetSpeedRotation(unit):
    speed = 0
    rotation = 0

    if unit["speed"] == "BACKWARD":
        speed = -1
    elif unit["speed"] == "NONE":
        speed = 0
    elif unit["speed"] == "FORWARD":
        speed = 1

    if unit["rotation"] == "SLOW_LEFT":
        rotation = -1
    elif unit["rotation"] == "NONE":
        rotation = 0
    elif unit["rotation"] == "SLOW_RIGHT":
        rotation = 1

    return speed, rotation


def GetState_Vojak(unit):
    speed, rotation = GetSpeedRotation(unit)

    if len(unit["opponentsInView"]) > 0:
        canSeeOponent = 1
    else:
        canSeeOponent = 0

    if canSeeOponent == 1:
        orientationToOponent = math_util.angle_between_unit_and_point(unit, unit["opponentsInView"][0]["x"],
                                                                      unit["opponentsInView"][0]["y"])
    else:
        orientationToOponent = 181

    if canSeeOponent == 1:
        distanceToOpnonet = math_util.distance(unit["x"], unit["y"], unit["opponentsInView"][0]["x"],
                                               unit["opponentsInView"][0]["y"])
    else:
        distanceToOpnonet = 31


    return speed, rotation, canSeeOponent, orientationToOponent, distanceToOpnonet


def GetState_Nabiralec(api, unit):
    speed, rotation = GetSpeedRotation(unit)

    if len(unit["resourcesInView"]) > 0:
        canSeeResource = 1

        closestResource = unit["resourcesInView"][0]
        distanceToResource = math_util.distance(unit["x"], unit["y"], closestResource["x"], closestResource["y"])

        for resource in unit["resourcesInView"]:
            if math_util.distance(unit["x"], unit["y"], resource["x"], resource["y"]) < distanceToResource:
                closestResource = resource
                distanceToResource = math_util.distance(unit["x"], unit["y"], closestResource["x"],
                                                        closestResource["y"])

        orientationToResource = math_util.angle_between_unit_and_point(unit, closestResource["x"], closestResource["y"])
        distanceToResource = math_util.distance(unit["x"], unit["y"], closestResource["x"], closestResource["y"])
    else:
        canSeeResource = 0
        orientationToResource = 31
        distanceToResource = 31

    distanceToObsticle = GetDistanceToWall(api, unit)

    if distanceToObsticle > 50:
        distanceToObsticle = 50

    return speed, rotation, canSeeResource, orientationToResource, distanceToResource, distanceToObsticle
    #(2,        1,      1,      30,                         2,              24,             4)
    #naprej,    desno,  vidim,  naravnost, proti resourcu,  blizu resourca, dle od stene,   naredim: rotacijo na none


def GetReward_Vojak(state):
    if state[0] == 1 and state[2] == 1 and abs(state[3]) < 5:
        return 1

    return 0


def GetReward_Nabiralec(api, unit, state):
    #print("A", unit["id"], state, botWasCloseToFood[unit["id"]], lastPobranih, pobranih)

    if botWasCloseToFood[unit["id"]] and lastPobranih < pobranih:
        api.say_something(unit["id"], "FAJN JE BLO")
        print("A", "NAŽRO SM SE GA KO PRASICA")
        return 10000

    botWasCloseToFood[unit["id"]] = False

    #if state[2] == 1:# and unit["id"] == 1:
    #print("A", unit["id"], state)


    if state[5] < 5:
        #api.say_something(unit["id"], "STENAAAAAA")
        #print("A", "BLIZU STENE")
        return -50

    if state[0] == 0 and state[1] == 0:
        #print("A", "NE PREMIKA")
        return -50

    if state[0] == 1 and state[2] == 1 and abs(state[3]) < 22 and state[4] < 3:
        botWasCloseToFood[unit["id"]] = True
        print("A", unit["id"], "NJAM NJAM")
        api.say_something(unit["id"], "FUTER LMAO")
        return 100
    elif state[0] == 0 and state[2] == 1 and abs(state[3]) < 22 and state[4] < 3:
        return -1

    elif state[0] == 1 and state[2] == 1 and abs(state[3]) < 18 and state[4] < 6:
        print("A", "NJAM")
        api.say_something(unit["id"], "FUTER BLIZO")
        return 10
    elif state[0] == 0 and state[2] == 1 and abs(state[3]) < 18 and state[4] < 6:
        return -1

    elif state[0] == 1 and state[2] == 1 and abs(state[3]) < 15 and state[4] < 15:
        print("A", "N")
        #api.say_something(unit["id"], "FUTER VIDIM")
        return 1
    elif state[0] == 0 and state[2] == 1 and abs(state[3]) < 15 and state[4] < 15:
        return -1

    return -5


def DoActionVojak(api, unit, action):
    if random.random() < 0.01:
        action = random.randint(0, 6)

    # Move randomly, turnLeft, stopTurning, turnRight, stopMoving, moveForward, Shoot

    if action == 0:

        # If the unit is not going anywhere, we send it
        # to a random valid location on the map.
        if len(unit["navigationPath"]) == 0:

            # Generate new x and y until you get a position on the map
            # where there is no obstacle.
            while True:
                x = random.randint(0, constants.MAP_WIDTH - 1)
                y = random.randint(0, constants.MAP_HEIGHT - 1)

                # If map[x][y] equals false it means that at (x,y) there is no obstacle.
                if constants.MAP[x][y] is False:
                    # Send the unit to (x, y)
                    api.navigation_start(unit["id"], x, y)
                    break



    elif action == 1:
        api.set_rotation(unit["id"], "SLOW_LEFT")

    elif action == 2:
        api.set_rotation(unit["id"], "NONE")

    elif action == 3:
        api.set_rotation(unit["id"], "SLOW_RIGHT")

    elif action == 4:
        api.set_speed(unit["id"], "NONE")

    elif action == 5:
        api.set_speed(unit["id"], "FORWARD")

    elif action == 6:
        api.shoot(unit["id"])

def DoActionNabiralec(api, unit, action):
    if random.random() < 0.01:
        action = random.randint(0, 5)

    # Move randomly, stopMoving, moveForward, turnLeft, stopTurning, turnRight

    if action == 0:

        # If the unit is not going anywhere, we send it
        # to a random valid location on the map.
        if len(unit["navigationPath"]) == 0:

            # Generate new x and y until you get a position on the map
            # where there is no obstacle.
            while True:
                x = random.randint(0, constants.MAP_WIDTH - 1)
                y = random.randint(0, constants.MAP_HEIGHT - 1)

                # If map[x][y] equals false it means that at (x,y) there is no obstacle.
                if constants.MAP[x][y] is False:
                    # Send the unit to (x, y)
                    api.navigation_start(unit["id"], x, y)
                    break

    elif action == 1:
        api.set_speed(unit["id"], "NONE")

    elif action == 2:
        api.set_speed(unit["id"], "FORWARD")

    elif action == 3:
        api.set_rotation(unit["id"], "SLOW_LEFT")

    elif action == 4:
        api.set_rotation(unit["id"], "NONE")

    elif action == 5:
        api.set_rotation(unit["id"], "SLOW_RIGHT")


old_discrete_state_of_units = dict()

# Initial implementation keeps picking random locations on the map
# and sending units there. Worker units collect resources if they
# see them while warrior units shoot if they see opponents.
class MyBot(Bot):
    # This method is called 10 times per game second and holds current
    # game state. Use Api object to call actions on your units.
    # - GameState reference: https://docs.liagame.com/api/#gamestate
    # - Api reference:       https://docs.liagame.com/api/#api-object
    def update(self, state, api):
        global VOJAK_q_table
        global NABIRALEC_q_table

        global lastFood
        global lastPobranih
        global pobranih

        global a

        if state["resources"] > lastFood:
            pobranih += 1
            print("A", "----------------------------------------------------------------------------", pobranih)



        #print("A", constants.VIEWING_AREA_LENGTH)
        #print("A", constants.VIEWING_AREA_WIDTH)

        #print("A", )
        #print("A", state["resources"])
        #print("A", constants.VIEWING_AREA_LENGTH)
        #print("A", constants.VIEWING_AREA_WIDTH)

        # If you have enough resources to spawn a new warrior unit then spawn it.
        if state["resources"] >= constants.WARRIOR_PRICE:
            #api.spawn_unit(UnitType.WARRIOR)
            api.spawn_unit(UnitType.WORKER)

        # We iterate through all of our units that are still alive.
        for unit in state["units"]:

            '''
            if unit["id"] == 1:
                if len(unit["navigationPath"]) == 0:
                    if a:
                        api.navigation_start(unit["id"], 25, 50)
                        a = False
                    else:
                        #api.navigation_start(unit["id"], 35, 50)
                        print("A", unit["orientationAngle"])
                        if not(215 < unit["orientationAngle"] < 260):
                            api.set_rotation(unit["id"], "SLOW_RIGHT")
                            GetDistanceToWall(api, unit)
                        else:
                            api.set_rotation(unit["id"], "NONE")
                            GetDistanceToWall(api, unit)
                return
            '''

            if unit["type"] == "WARRIOR":
                stateOfUnit = GetState_Vojak(unit)
                reward = GetReward_Vojak(stateOfUnit)

                if unit["id"] not in old_discrete_state_of_units.keys():
                    discrete_state = Get_discrete_state_Vojak(np.array(stateOfUnit))
                    action = GetAction_Vojak(discrete_state)
                    DoActionVojak(api, unit, action)
                    old_discrete_state_of_units[unit["id"]] = discrete_state
                    continue
                else:
                    discrete_state = old_discrete_state_of_units[unit["id"]]

                action = GetAction_Vojak(discrete_state)

                new_discrete_state = Get_discrete_state_Vojak(np.array(stateOfUnit))
                max_future_q = np.max(VOJAK_q_table[new_discrete_state])
                current_q = VOJAK_q_table[discrete_state + (action,)]

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                VOJAK_q_table[discrete_state + (action,)] = new_q
                old_discrete_state_of_units[unit["id"]] = new_discrete_state

                action = GetAction_Vojak(new_discrete_state)
                DoActionVojak(api, unit, action)

            else:
                #print("A", sha1(NABIRALEC_q_table).hexdigest())

                stateOfUnit = GetState_Nabiralec(api, unit)
                reward = GetReward_Nabiralec(api, unit, stateOfUnit)

                if unit["id"] not in old_discrete_state_of_units.keys():
                    discrete_state = Get_discrete_state_Nabiralec(np.array(stateOfUnit))
                    action = GetAction_Nabiralec(discrete_state)
                    DoActionNabiralec(api, unit, action)
                    old_discrete_state_of_units[unit["id"]] = discrete_state
                    continue
                else:
                    discrete_state = old_discrete_state_of_units[unit["id"]]

                action = GetAction_Nabiralec(discrete_state)

                new_discrete_state = Get_discrete_state_Nabiralec(np.array(stateOfUnit))
                max_future_q = np.max(NABIRALEC_q_table[new_discrete_state])
                current_q = NABIRALEC_q_table[discrete_state + (action,)]

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                NABIRALEC_q_table[discrete_state + (action,)] = new_q

                old_discrete_state_of_units[unit["id"]] = new_discrete_state
                action = GetAction_Nabiralec(new_discrete_state)
                DoActionNabiralec(api, unit, action)

        lastPobranih = pobranih
        lastFood = state["resources"]



# Connects your bot to Lia game engine, don't change it.
if __name__ == "__main__":
    global VOJAK_q_table
    global NABIRALEC_q_table

    # VOJAK
    VOJAK_NUM_OF_ACTIONS = 7 # Move randomly, turnLeft, stopTurning, turnRight, stopMoving, moveForward, Shoot

    # speed
    # rotation
    # canSeeOponent
    # orientation to oponent
    # distance to oponent
    VOJAK_DISCRETE_OS_SIZE = [                  3,      3,      2,      361+1,      30+1+1]

    VOJAK_OBSERVATION_SPACE_LOW = np.array([    -1,     -1,     0,      -180,       0])
    VOJAK_OBSERVATION_SPACE_HIGH = np.array([   1+1,    1+1,    1+1,    181+1,      30+1+1])

    # Q-Learning settings
    #LEARNING_RATE = 0.05
    #DISCOUNT = 0.85

    VOJAK_discrete_os_win_size = (VOJAK_OBSERVATION_SPACE_HIGH - VOJAK_OBSERVATION_SPACE_LOW) / VOJAK_DISCRETE_OS_SIZE
    #print("A", VOJAK_discrete_os_win_size)

    if os.path.isfile('VOJAK_qTable.npy'):
        print("A", "Vojak loaded")
        VOJAK_q_table = np.load('VOJAK_qTable.npy', allow_pickle=True)
    else:
        VOJAK_q_table = np.random.uniform(low=-2, high=0, size=(VOJAK_DISCRETE_OS_SIZE + [VOJAK_NUM_OF_ACTIONS]))


    #NABIRALEC
    NABIRALEC_NUM_OF_ACTIONS = 6  # Move randomly, stopMoving, moveForward, turnLeft, stopTurning, turnRight

    # speed
    # rotation
    # canSeefood
    # orientation to food
    # distance to food
    # distance to obsticle
    NABIRALEC_DISCRETE_OS_SIZE =                [3,      3,     2,         60+1+1,     30+1+1,   25]

    NABIRALEC_OBSERVATION_SPACE_LOW = np.array( [-1,     -1,    0,         -30,        0,          0])
    NABIRALEC_OBSERVATION_SPACE_HIGH = np.array([1+1,    1+1,   1+1,        30+1+1,    30+1+1,     50+1])

    # Q-Learning settings
    LEARNING_RATE = 0.01
    DISCOUNT = 0.01

    NABIRALEC_discrete_os_win_size = (NABIRALEC_OBSERVATION_SPACE_HIGH - NABIRALEC_OBSERVATION_SPACE_LOW) / NABIRALEC_DISCRETE_OS_SIZE
    #print("A", NABIRALEC_discrete_os_win_size)

    if os.path.isfile('NABIRALEC_qTable.npy'):
        print("A", "Nabiralec loaded")
        NABIRALEC_q_table = np.load('NABIRALEC_qTable.npy', allow_pickle=True)
    else:
        NABIRALEC_q_table = np.random.uniform(low=-2, high=0, size=(NABIRALEC_DISCRETE_OS_SIZE + [NABIRALEC_NUM_OF_ACTIONS]))

    #print("A", sha1(NABIRALEC_q_table).hexdigest())

    #FindMaxInArray(NABIRALEC_q_table)
    #print("A", "max value in arr pametn:", np.unravel_index(NABIRALEC_q_table.argmax(), NABIRALEC_q_table.shape),
    #      NABIRALEC_q_table.shape)
    # print("A", sha1(NABIRALEC_q_table).hexdigest())

    asyncio.get_event_loop().run_until_complete(connect(MyBot()))


    print("A", "pametn1:", pobranih)

    with open('Results.txt', 'a') as file:
        file.write(str(pobranih) + "\n")

    np.save("VOJAK_qTable.npy", VOJAK_q_table, allow_pickle=True)
    print("A", "Vojak saved")

    np.save("NABIRALEC_qTable.npy", NABIRALEC_q_table, allow_pickle=True)
    print("A", "Nabiralec saved")

    SaveIfBetter(VOJAK_q_table, "VOJAK", pobranih)
    SaveIfBetter(NABIRALEC_q_table, "NABIRALEC", pobranih)