from gym_env.envs.Intersection import TwoWayIntersection
import os
import sys
import optparse
import random
from sumolib import checkBinary
import traci
import time
import numpy as np

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
        self.sim_end = 100 #simulation end time
        self.run_flag = False #running simulation flag -> if True - simulation is running
        self.sim_step = 0 #represents each step of simulation
        self.delay = 1 #Delay between epizodes
        # self.agents = [TwoWayIntersection(id="0"), TwoWayIntersection(id="1")] #list of intersectons in network
        self.agents = [TwoWayIntersection(id="0"), TwoWayIntersection(id="1")]
        self.multi_det = "det0"
        # configure spaces for multiple intersections in network (Multiple agents for each intersection)
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            self.action_space.append(spaces.Discrete(len(agent.actions)))
            light_space = spaces.Discrete(len(agent.actions))
            obs_space = spaces.Box(np.array([0 for i in range(len(agent.dets))]), np.array([8 for i in range(len(agent.dets))]), dtype=np.int)
            # self.observation_space.append(spaces.Box(np.array([0 for i in range(len(agent.dets))]), np.array([8 for i in range(len(agent.dets))]), dtype=np.int))
            all_obs = spaces.Tuple((obs_space, light_space))
            self.observation_space.append(obs_space)

        # Paths for SUMO configuration files
        self.file_path = os.path.join(os.path.dirname(__file__), "cfg_2")  # Base path to xml project files
        # self.file_path = os.path.join(os.path.dirname(__file__), "cfg_1")  # Base path to xml project files
        self.roufile = os.path.join(self.file_path, "Intersection.rou.xml")  # Name of routes file
        self.cfgfile = os.path.join(self.file_path, "Intersection.sumocfg")  # Name of configuration file
        self.infofile = os.path.join(self.file_path, "tripinfo.xml")  # name of SUMO output file




    def generateRoutes(self): #method generate routes (different scenarios), write to file "Intersection.rou.xml"
        random.seed(1)  # make tests reproducible
        N = 50  # number of vehicles
        pWE = 1. / 5
        pEW = 1. / 10
        pNS = 1. / 15
        pSN = 1. / 20
        with open(self.roufile, "w") as routes:
            print("""<routes>
                <vType accel="2.6" decel="4.5" id="Car" length="4.0" maxSpeed="70.0" sigma="0.4" guiShape="passenger" />
                <route id="right" edges="W_0 0_1 1_E2" />
                <route id="left" edges="E2_1 1_0 0_W" />
                <route id="down" edges="N_0 0_S" />
                <route id="up" edges="S_0 0_N" />""", file=routes)
            vehNr = 0
            # First scenario - 10 cars from West to East
            for i in range(N):
                print('    <vehicle id="right_%i" type="Car" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            # for i in range(N):
            #     if random.uniform(0, 1) < pWE:
            #         print('    <vehicle id="right_%i" type="Car" route="right" depart="%i" />' % (
            #             vehNr, i), file=routes)
            #         vehNr += 1
            #     if random.uniform(0, 1) < pEW:
            #         print('    <vehicle id="left_%i" type="Car" route="left" depart="%i" />' % (
            #             vehNr, i), file=routes)
            #         vehNr += 1
            #     if random.uniform(0, 1) < pNS:
            #         print('    <vehicle id="down_%i" type="Car" route="down" depart="%i" />' % (
            #             vehNr, i), file=routes)
            #         vehNr += 1
            #     if random.uniform(0, 1) < pSN:
            #         print('    <vehicle id="up_%i" type="Car" route="up" depart="%i" />' % (
            #             vehNr, i), file=routes)
            #         vehNr += 1
            print("</routes>", file=routes)

    # def generateRoutes(self): #method generate routes (different scenarios), write to file "Intersection.rou.xml"
    #     random.seed(1)  # make tests reproducible
    #     N = 50  # number of vehicles
    #     pWE = 1. / 5
    #     pEW = 1. / 10
    #     pNS = 1. / 15
    #     pSN = 1. / 20
    #     with open(self.roufile, "w") as routes:
    #         print("""<routes>
    #             <vType accel="2.6" decel="4.5" id="Car" length="4.0" maxSpeed="70.0" sigma="0.4" guiShape="passenger" />
    #             <route id="right" edges="W_0 0_1" />
    #             <route id="down" edges="N_0 0_S" />
    #             <route id="up" edges="S_0 0_N" />""", file=routes)
    #         vehNr = 0
    #         # First scenario - 10 cars from West to East
    #         for i in range(N):
    #             print('    <vehicle id="right_%i" type="Car" route="right" depart="%i" />' % (
    #                 vehNr, i), file=routes)
    #             vehNr += 1
    #         # for i in range(N):
    #         #     if random.uniform(0, 1) < pWE:
    #         #         print('    <vehicle id="right_%i" type="Car" route="right" depart="%i" />' % (
    #         #             vehNr, i), file=routes)
    #         #         vehNr += 1
    #         #     if random.uniform(0, 1) < pEW:
    #         #         print('    <vehicle id="left_%i" type="Car" route="left" depart="%i" />' % (
    #         #             vehNr, i), file=routes)
    #         #         vehNr += 1
    #         #     if random.uniform(0, 1) < pNS:
    #         #         print('    <vehicle id="down_%i" type="Car" route="down" depart="%i" />' % (
    #         #             vehNr, i), file=routes)
    #         #         vehNr += 1
    #         #     if random.uniform(0, 1) < pSN:
    #         #         print('    <vehicle id="up_%i" type="Car" route="up" depart="%i" />' % (
    #         #             vehNr, i), file=routes)
    #         #         vehNr += 1
    #         print("</routes>", file=routes)

    def getOptions(self): #Method returns simulation options like run with gui or without
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--nogui", action="store_true",
                              default=True, help="run the commandline version of sumo")
        options, args = opt_parser.parse_args()
        return options

    def startSumo(self):
        if not self.run_flag:
            options = self.getOptions()
            if options.nogui:
                sumoBinary = checkBinary('sumo')
            else:
                sumoBinary = checkBinary('sumo-gui')
            # Generate route file
            self.generateRoutes()
            # Start SUMO as subprocess, connect python script and run
            traci.start([sumoBinary, "-c", self.cfgfile, "--tripinfo-output", self.infofile])
            self.sim_step = 0
            self.run_flag = True
        else:
            print("Nie udalo sie uruchomic symulacji!")

    def stopSumo(self):
        if self.run_flag:
            traci.close()
            sys.stdout.flush()
            self.run_flag = False
        else:
            print("Symulacja nie jest wlaczona!")


    def get_reward(self, agent):
        reward = 0
        for det in agent.dets:
            o = traci.multientryexit.getLastStepVehicleNumber(det)
            if o > 6:
                reward += -10
            else:
                reward += 1
        return reward

    def get_observation(self, agent):
        observation = []
        for det in agent.dets:
            observation.append(traci.multientryexit.getLastStepVehicleNumber(det))
        return tuple(observation)


    def step(self, action_n):
        observation_n = []
        reward_n = []
        self.sim_step += 1 #increase simulation step
        self.applyAction(action_n)
        traci.simulationStep()
        for agent in self.agents:
            observation_n.append(self.get_observation(agent))
            reward_n.append(self.get_reward(agent))
            # reward_n.append(self.rew())
        end = self.sim_step >= self.sim_end
        if end:
            self.stopSumo()
        return observation_n, reward_n, end, {}

    def reset(self): #Method define steps to reset simulation
        observation_n = []
        self.stopSumo() #Stop running simulation
        time.sleep(self.delay/100.0) #Wait 10 ms (to stop all SUMO processes)
        self.startSumo() #Start simulation again
        for agent in self.agents:
            observation_n.append(self.get_observation(agent))
        return observation_n

    def applyAction(self, action):
        for act, lgt in zip(action, self.agents): #set light (for loop if in the future will be more intersections)
            signal = lgt.setLight(act)
            traci.trafficlight.setPhase(lgt.intersection_id, signal)

    def rew(self):
        s = traci.multientryexit.getLastStepMeanSpeed(self.multi_det)
        c = traci.multientryexit.getLastStepVehicleNumber(self.multi_det)
        r = max(s * c, 0)
        return r


