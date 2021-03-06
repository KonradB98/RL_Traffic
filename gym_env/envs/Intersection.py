#Class define overall structure of Intersections
class Intersection:
    def __init__(self, id, actions):
        self.state = 0 #Actual intersection state (id of light phase)
        self.step = 0 #Auxiliary variable for duration time
        self.intersection_id = id #Id of intersection
        self.actions = actions #Action (id of light phase) implemented by Agent

    def setLight(self, action):
        avl_act = self.available_actions(action)
        if action != self.state and avl_act:
            self.state = action
            self.step = 0
        else:
            self.step += 1
        return self.actions[self.state]

    def getState(self):
        return self.state

    def getActions(self):
        return self.actions

#Subclass that defines simple, two way intersection
class TwoWayIntersection(Intersection):
    def __init__(self, id, lanes = []):
        self.duration = 1 #Duration time
        self.dets = ["detector" + id + "_{}".format(i) for i in range(4)]
        self.induction_loops = ["mloop{}".format(i) for i in range(8)]
        self.lanes = lanes
        super(TwoWayIntersection, self).__init__(id=id, actions=[0, 1, 2, 3])


    def available_actions(self, action):
        if self.state == 0:
            return action == 1
        elif self.state == 2:
            return action == 3
        elif self.state == 1:
            return action == 2 and self.step > self.duration
        elif self.state == 3:
            return action == 0 and self.step > self.duration
        else:
            print("ERROR")
