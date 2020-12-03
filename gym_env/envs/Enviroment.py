from gym_env.envs.Intersection import TwoWayIntersection
import os
import sys
import optparse
import random
from sumolib import checkBinary
import traci
import time
import numpy as np
import math

import gym
from gym import error, spaces, utils

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

class TrafficEnv(gym.Env): #Class define structure od SUMO enviroment

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self): #Constructor
        self.sim_end = 1000 #simulation end time
        self.run_flag = False #running simulation flag -> if True - simulation is running
        self.sim_step = 0 #represents each step of simulation
        self.delay = 1 #Delay between epizodes
        self.multi_det = "det0"
        self.seeding = 0
        self.lanes0 = ["W_0_0", "S_0_0", "N_0_0", "1_0_0"]
        self.lanes1 = ["0_W_0", "0_S_0", "0_N_0", "0_1_0"]

        self.agent = TwoWayIntersection(id="0", lanes=self.lanes0)
        # self.agents = [TwoWayIntersection(id="0", lanes=self.lanes0), TwoWayIntersection(id="1", lanes=self.lanes1)]

        self.action_space = spaces.Discrete(len(self.agent.actions))
        lgt = [spaces.Discrete(len(self.agent.actions))]
        # obs = spaces.Box(np.array([0 for i in range(len(self.agent.dets))]), np.array([30 for i in range(len(self.agent.dets))]), dtype=np.int)
        # obs = spaces.Box(low=float('-inf'), high=float('inf'), shape=(len(self.agent.lanes) * 4,))
        obs = spaces.Box(low=float('-inf'), high=float('inf'), shape=(len(self.agent.lanes) * 5,))
        self.observation_space = obs

        # self.action_space = []
        # self.observation_space = []
        # for agent in self.agents:
        #     self.action_space.append(spaces.Discrete(len(agent.actions)))
        #     obs = spaces.Box(low=float('-inf'), high=float('inf'), shape=(len(agent.lanes) * 5,))
        #     self.observation_space.append(obs)


        self.file_path = os.path.join(os.path.dirname(__file__), "cfg_1")  # Base path to xml project files
        # self.file_path = os.path.join(os.path.dirname(__file__), "cfg_2")  # Base path to xml project files

        self.roufile = os.path.join(self.file_path, "Intersection.rou.xml")  # Name of routes file
        self.cfgfile = os.path.join(self.file_path, "Intersection.sumocfg")  # Name of configuration file
        self.infofile = os.path.join(self.file_path, "tripinfo.xml")  # name of SUMO output file
        self.netfile = os.path.join(self.file_path, "Intersection.net.xml")
        self.addfile = os.path.join(self.file_path, "Intersection.add.xml")
        self.last_step_wt = 0
        self.wt = 0
        self.hv = 0
        self.ms = 0



    def getOptions(self): #Method returns simulation options like run with gui or without
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--nogui", action="store_true",
                              default=False, help="run the commandline version of sumo")
        options, args = opt_parser.parse_args()
        return options

    def startSumo(self):
        if not self.run_flag:
            options = self.getOptions()
            if options.nogui:
                sumoBinary = checkBinary('sumo')
            else:
                sumoBinary = checkBinary('sumo-gui')
            self.seeding += 1
            self.generate_scenario(self.seeding)
            # self.generate_simple_scenario(self.seeding)
            traci.start([sumoBinary, "--quit-on-end", "-c", self.cfgfile, "--tripinfo-output", self.infofile])
            self.sim_step = 0
            self.run_flag = True
            self.hv = self.ms = self.wt = 0
        else:
            print("Nie udalo sie uruchomic symulacji!")

    def stopSumo(self):
        if self.run_flag:
            traci.close()
            sys.stdout.flush()
            self.run_flag = False
            with open('statistics/HV.txt', 'a') as f:
                f.write("%f" % self.hv + "\n")
            with open('statistics/WT.txt', 'a') as f:
                f.write("%f" % self.wt + "\n")
            with open('statistics/MS.txt', 'a') as f:
                f.write("%f" % self.ms + "\n")
            # l = []
            # l.append(self.hv)
            # l.append(self.wt)
            # l.append(self.ms)
            # with open('stats.txt', 'a') as f:
            #     # fo.write('\n'.join([' '.join(i) for i in l]))
            #     for item in l:
            #         f.write("%f" % item + " ")
            #     f.write("\n")
            self.hv = self.ms = self.wt = 0
        else:
            print("Symulacja nie jest wlaczona!")

    def check_detector(self, det):
        count = traci.multientryexit.getLastStepVehicleNumber(det)
        light = self.agent.state
        return count


    def get_reward(self):
        reward = 0
        for lane in self.agent.lanes:
            reward -= traci.lane.getLastStepHaltingNumber(lane)

        # for lane in self.agent.lanes:
        #     reward -= traci.lane.getWaitingTime(lane)


        # for lane in self.agent.lanes:
        #     r1 = traci.lane.getLastStepHaltingNumber(lane)
        #     r2 = traci.lane.getLastStepMeanSpeed(lane)
        #     reward += r2 - 2*r1

        # for lane in self.agent.lanes:
        #     r1 = traci.lane.getLastStepMeanSpeed(lane)
        #     r2 = traci.lane.getWaitingTime(lane)
        #     reward += 2*r1 - r2

        # reward = 1
        return reward

    def get_observation(self):
        observation = []

        ##-------------5 parameters observation-----------------##

        for lane in self.agent.lanes:
            observation.append(traci.lane.getLastStepVehicleNumber(lane))
            observation.append(traci.lane.getLastStepMeanSpeed(lane))
            observation.append(traci.lane.getWaitingTime(lane))
            observation.append(traci.lane.getLastStepHaltingNumber(lane))
            observation.append(self.agent.state)

        ##-------------4 parameters observation-----------------##

        # for lane in self.agent.lanes:
        #     observation.append(traci.lane.getLastStepVehicleNumber(lane))
        #     observation.append(traci.lane.getLastStepMeanSpeed(lane))
        #     observation.append(traci.lane.getWaitingTime(lane))
        #     observation.append(self.agent.state)


        obs = np.array(observation)

        return obs

        ##-----------------------Q-learning--------------------------##

        # for lane in self.agent.lanes:
        #     observation.append(traci.lane.getLastStepHaltingNumber(lane))
        # return tuple(observation)


    def step(self, action):
        self.sim_step += 1 #increase simulation step
        self.applyAction(action)
        traci.simulationStep()
        observation = self.get_observation()
        reward = self.get_reward()
        self.stats()
        # print(observation)
        # print(reward)
        # for lane in self.agent.lanes:
        #     jam = traci.lane.getLastStepHaltingNumber(lane)
        #     if jam >= 5:
        #         end = True
        #         break
        #     else:
        #         end = False
        done = self.sim_step >= self.sim_end
        if done:
            self.stopSumo()
        return observation, reward, done, {}

    def reset(self): #Method define steps to reset simulation
        self.stopSumo() #Stop running simulation
        time.sleep(self.delay/100.0) #Wait 10 ms (to stop all SUMO processes)
        self.startSumo() #Start simulation again
        self.last_step_wt = 0
        observation = self.get_observation()
        return observation

    def applyAction(self, action):
        signal = self.agent.setLight(action)
        traci.trafficlight.setPhase(self.agent.intersection_id, signal)

    def stats(self):
        for lane in self.agent.lanes:
            self.hv += traci.lane.getLastStepHaltingNumber(lane)
            self.wt += traci.lane.getWaitingTime(lane)
            self.ms += traci.lane.getLastStepMeanSpeed(lane)


    # -----------------------------------------------------------------------#
    # -------------------------2 INTERSECTIONS-------------------------------#
    # -----------------------------------------------------------------------#

    # def get_reward(self, agent):
    #     reward = 0
    #     for lane in agent.lanes:
    #         reward -= traci.lane.getLastStepHaltingNumber(lane)
    #     return reward
    #
    # def get_observation(self, agent):
    #     observation = []
    #
    #     ##-------------5 parameters observation-----------------##
    #
    #     for lane in agent.lanes:
    #         observation.append(traci.lane.getLastStepVehicleNumber(lane))
    #         observation.append(traci.lane.getLastStepMeanSpeed(lane))
    #         observation.append(traci.lane.getWaitingTime(lane))
    #         observation.append(traci.lane.getLastStepHaltingNumber(lane))
    #         observation.append(agent.state)
    #
    #     obs = np.array(observation)
    #
    #     return obs
    #
    # def step(self, action_n):
    #     observation_n = []
    #     reward_n = []
    #     self.sim_step += 1 #increase simulation step
    #     self.applyAction(action_n)
    #     traci.simulationStep()
    #     for agent in self.agents:
    #         observation_n.append(self.get_observation(agent))
    #         reward_n.append(self.get_reward(agent))
    #
    #     done = self.sim_step >= self.sim_end
    #     if done:
    #         self.stopSumo()
    #     return observation_n, reward_n, done, {}
    #
    # def reset(self): #Method define steps to reset simulation
    #     observation_n = []
    #     self.stopSumo() #Stop running simulation
    #     time.sleep(self.delay/100.0) #Wait 10 ms (to stop all SUMO processes)
    #     self.startSumo() #Start simulation again
    #     for agent in self.agents:
    #         observation_n.append(self.get_observation(agent))
    #     return observation_n
    #
    # def applyAction(self, action):
    #     for act, lgt in zip(action, self.agents): #set light (for loop if in the future will be more intersections)
    #         signal = lgt.setLight(act)
    #         traci.trafficlight.setPhase(lgt.intersection_id, signal)



    # -----------------------------------------------------------------------#
    # -----------------------------Scenarios---------------------------------#
    # -----------------------------------------------------------------------#

    def generate_scenario(self, seed):  # method generate routes (different scenarios), write to file "Intersection.rou.xml"
        # random.seed(1)  # make tests reproducible
        np.random.seed(seed)
        N = 200  # number of vehicles

        t = np.random.weibull(2, N)
        t = np.sort(t)

        cgs = []
        min_o = math.floor(t[1])
        max_o = math.floor(t[-1])
        min_n = 0
        max_n = self.sim_end

        for i in t:
            cgs = np.append(cgs, ((max_n - min_n) / (max_o - min_o)) * (i - max_o) + max_n)

        cgs = np.rint(cgs)

        with open(self.roufile, "w") as routes:
            print("""<routes>
                <vType accel="2.6" decel="4.5" id="Car" length="4.0" maxSpeed="70.0" sigma="0.4" guiShape="passenger" />
                <route id="right" edges="W_0 0_1" />
                <route id="down" edges="N_0 0_S" />
                <route id="left" edges="1_0 0_W" />
                <route id="rd" edges="W_0 0_S" />
                <route id="ur" edges="S_0 0_1" />
                <route id="lu" edges="1_0 0_N" />
                <route id="dl" edges="N_0 0_W" />
                <route id="up" edges="S_0 0_N" />""", file=routes)

            for cc, step in enumerate(cgs):
                sot = np.random.uniform()
                if sot < 0.8:
                    rs = np.random.randint(1, 5)
                    if rs == 1:
                        print('    <vehicle id="right_%i" type="Car" route="right" depart="%s" />' % (
                            cc, step), file=routes)
                    elif rs == 2:
                        print('    <vehicle id="left_%i" type="Car" route="left" depart="%s" />' % (
                            cc, step), file=routes)
                    elif rs == 3:
                        print('    <vehicle id="up_%i" type="Car" route="up" depart="%s" />' % (
                            cc, step), file=routes)
                    else:
                        print('    <vehicle id="down_%i" type="Car" route="down" depart="%s" />' % (
                            cc, step), file=routes)
                else:
                    rt = np.random.randint(1, 5)
                    if rt == 1:
                        print('    <vehicle id="rd_%i" type="Car" route="rd" depart="%s" />' % (
                            cc, step), file=routes)
                    elif rt == 2:
                        print('    <vehicle id="ur_%i" type="Car" route="ur" depart="%s" />' % (
                            cc, step), file=routes)
                    elif rt == 3:
                        print('    <vehicle id="lu_%i" type="Car" route="lu" depart="%s" />' % (
                            cc, step), file=routes)
                    else:
                        print('    <vehicle id="dl_%i" type="Car" route="dl" depart="%s" />' % (
                            cc, step), file=routes)

            print("</routes>", file=routes)

    def generate_simple_scenario(self, seed):  # method generate routes (different scenarios), write to file "Intersection.rou.xml"
        # random.seed(1)  # make tests reproducible
        np.random.seed(seed)
        N = 50  # number of vehicles

        t = np.random.weibull(2, N)
        t = np.sort(t)

        cgs = []
        min_o = math.floor(t[1])
        max_o = math.floor(t[-1])
        min_n = 0
        max_n = self.sim_end

        for i in t:
            cgs = np.append(cgs, ((max_n - min_n) / (max_o - min_o)) * (i - max_o) + max_n)

        cgs = np.rint(cgs)

        with open(self.roufile, "w") as routes:
            print("""<routes>
                <vType accel="2.6" decel="4.5" id="Car" length="4.0" maxSpeed="70.0" sigma="0.4" guiShape="passenger" />
                <route id="right" edges="W_0 0_1" />
                <route id="down" edges="N_0 0_S" />
                <route id="left" edges="1_0 0_W" />
                <route id="rd" edges="W_0 0_S" />
                <route id="ur" edges="S_0 0_1" />
                <route id="lu" edges="1_0 0_N" />
                <route id="dl" edges="N_0 0_W" />
                <route id="up" edges="S_0 0_N" />""", file=routes)

            for cc, step in enumerate(cgs):
                print('    <vehicle id="right_%i" type="Car" route="right" depart="%s" />' % (
                            cc, step), file=routes)

            print("</routes>", file=routes)


##---------------------------------------------2 INTERSECTIONS---------------------------------------------------##

    def generate_scenario2(self, seed):  # method generate routes (different scenarios), write to file "Intersection.rou.xml"
        np.random.seed(seed)
        N = 300  # number of vehicles

        t = np.random.weibull(2, N)
        t = np.sort(t)

        cgs = []
        min_o = math.floor(t[1])
        max_o = math.floor(t[-1])
        min_n = 0
        max_n = self.sim_end

        for i in t:
            cgs = np.append(cgs, ((max_n - min_n) / (max_o - min_o)) * (i - max_o) + max_n)

        cgs = np.rint(cgs)

        with open(self.roufile, "w") as routes:
            print("""<routes>
                <vType accel="2.6" decel="4.5" id="Car" length="4.0" maxSpeed="70.0" sigma="0.4" guiShape="passenger" />
                <route id="right" edges="W_0 0_1 1_E2" />
                <route id="left" edges="E2_1 1_0 0_W" />
                <route id="down" edges="N_0 0_S" />
                <route id="up" edges="S_0 0_N" />
                <route id="down2" edges="N2_1 1_S2" />
                <route id="up2" edges="S2_1 1_N2" />""", file=routes)

            for cc, step in enumerate(cgs):
                sot = np.random.uniform()
                if sot < 0.7:
                    rs = np.random.randint(1, 3)
                    if rs == 1:
                        print('    <vehicle id="right_%i" type="Car" route="right" depart="%s" />' % (
                            cc, step), file=routes)
                    elif rs == 2:
                        print('    <vehicle id="left_%i" type="Car" route="left" depart="%s" />' % (
                            cc, step), file=routes)
                else:
                    rt = np.random.randint(1, 5)
                    if rt == 1:
                        print('    <vehicle id="up_%i" type="Car" route="up" depart="%s" />' % (
                            cc, step), file=routes)
                    elif rt == 2:
                        print('    <vehicle id="down_%i" type="Car" route="down" depart="%s" />' % (
                            cc, step), file=routes)
                    elif rt == 3:
                        print('    <vehicle id="up2_%i" type="Car" route="up2" depart="%s" />' % (
                            cc, step), file=routes)
                    else:
                        print('    <vehicle id="down2_%i" type="Car" route="down2" depart="%s" />' % (
                            cc, step), file=routes)

            print("</routes>", file=routes)





