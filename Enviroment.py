from Intersection import TwoWayIntersection
import os
import sys
import optparse
import random
from sumolib import checkBinary
import traci
import numpy as np
import time

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

class TrafficEnv: #Class define structure od SUMO enviroment
    def __init__(self): #Constructor
        self.detector = "detectors" #multi entry/exit detectors id
        self.intersections = [TwoWayIntersection(id="0")] #List of Lihts(intersections) -- now 1
        self.sim_end = 100 #simulation end time
        # self.obs_space = ["mloop{}".format(i) for i in range(8)] #induction loops detectors id's
        self.run_flag = False #running simulation flag -> if True - simulation is running
        self.sim_step = 0 #represents each step of simulation
        self.file_path = os.path.join(os.path.dirname(__file__), "cfg") #Base path to xml project files
        self.roufile = os.path.join(self.file_path, "Intersection.rou.xml") #Name of routes file
        self.cfgfile = os.path.join(self.file_path, "Intersection.sumocfg") #Name of configuration file
        self.infofile = os.path.join(self.file_path, "tripinfo.xml") #name of SUMO output file
        self.act_space = [a.actions for a in self.intersections] #Available actions for each intersections (intersections)
        self.delay = 1 #Delay between epizodes

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
                <route id="right" edges="W_0 0_E" />
                <route id="left" edges="E_0 0_W" />
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

    def getReward(self, o1):
        if o1 > 6:
            return -10
        else:
            return 1

    def getEnvState(self):
        veh_amount = traci.multientryexit.getLastStepVehicleNumber(self.detector)
        # veh_speed = traci.multientryexit.getLastStepMeanSpeed(self.detector)
        light_state = [lgt.state for lgt in self.intersections]
        return veh_amount, light_state

    def step(self, action):
        self.sim_step += 1 #increase simulation step
        for lgt in self.intersections: #set light (for loop if in the future will be more intersections)
            signal = lgt.setLight(action)
            traci.trafficlight.setPhase(lgt.intersection_id, signal)
        traci.simulationStep()
        o1, l = self.getEnvState()
        r = self.getReward(o1)
        end = self.sim_step > self.sim_end
        if end:
            self.stopSumo()
        return o1, r, end

    def stepik(self, action):
        self.sim_step += 1 #increase simulation step
        for act, lgt in zip(action, self.intersections): #set light (for loop if in the future will be more intersections)
            signal = lgt.setLight(act)
            traci.trafficlight.setPhase(lgt.intersection_id, signal)
        traci.simulationStep()
        o1, l = self.getEnvState()
        r = self.getReward(o1)
        end = self.sim_step > self.sim_end
        if end:
            self.stopSumo()
        return o1, r, end

    def reset(self): #Method define steps to reset simulation
        self.stopSumo() #Stop running simulation
        time.sleep(self.delay/100.0) #Wait 10 ms (to stop all SUMO processes)
        self.startSumo() #Start simulation again
        obs, l = self.getEnvState() #Get enviroment state
        return obs

    def getLightActions(self):
        actions = []
        for act in self.intersections:
            actions = act.getActions()
        return actions
