from Intersection import TwoWayIntersection
import os
import sys
import optparse
import random
from sumolib import checkBinary
import traci
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

class TrafficEnv: #Class define structure od SUMO enviroment
    # Constructor
    def __init__(self):
        self.detector = "detectors" #multi entry/exit detectors id
        self.lights = [TwoWayIntersection(id="0")] #List of Lihts(intersections) -- now 1
        self.sim_delay = 10 #time between simulations (episodes)
        self.sim_end = 100 #simulation end time
        self.obs_space = ["mloop{}".format(i) for i in range(8)] #induction loops detectors id's
        self.run_flag = False #running simulation flag -> if True - simulation is running
        self.sim_step = 0 #represents each step of simulation
        self.file_path = os.path.join(os.path.dirname(__file__), "cfg") #Base path to xml project files
        self.roufile = os.path.join(self.file_path, "Intersection.rou.xml") #Name of routes file
        self.cfgfile = os.path.join(self.file_path, "Intersection.sumocfg") #Name of configuration file
        self.infofile = os.path.join(self.file_path, "tripinfo.xml") #name of SUMO output file

    def generateRoutes(self): #method generate routes, write to file "Intersection.rou.xml"
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

    def getOptions(self):
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
        # return s1 * s2
        if o1 > 8:
            return -100
        else:
            return 1

    def observation(self):
        # xd = traci.vehicle.get
        veh_amount = traci.multientryexit.getLastStepVehicleNumber(self.detector)
        # veh_speed = traci.multientryexit.getLastStepMeanSpeed(self.detector)
        light_state = [lgt.state for lgt in self.lights]
        return veh_amount, light_state

    def getEnvState(self):
        state = []
        for det in self.obs_space:
            nb = traci.inductionloop.getLastStepVehicleNumber(det)
            ms = traci.inductionloop.getLastStepMeanSpeed(det)
            ld = traci.inductionloop.getTimeSinceDetection(det)
            pom = [nb, ms, ld]
            state.append(pom)
            # mv = traci.inductionloop.getLastStepVehicleNumber(det)
        state_mtrx = np.array(state)
        light_state = [lgt.state for lgt in self.lights]
        return state_mtrx, light_state

    def step(self, action):
        s = self.sim_step
        self.sim_step += 1 #increase simulation step
        for lgt in self.lights: #set light (for loop if in the future will be more intersections)
            signal = lgt.setLight(action)
            traci.trafficlight.setPhase(lgt.intersection_id, signal)
        # s, l = self.getEnvState()
        o1, l = self.observation()
        # print(q)
        r = self.getReward(o1)
        traci.simulationStep()
        # end = self.sim_step > self.sim_end
        # # s = self.sim_step
        # if end:
        #     self.stopSumo()
        return o1, r

    def getLightActions(self):
        actions = []
        for act in self.lights:
            actions = act.getActions()
        return actions