from dataclasses import dataclass
from typing import Sequence
import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import Point, LineString
import time

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

# PID controller
from dg_commons.controllers.pid import PID, PIDParam

# RRT* source:
# https://github.com/yrouben/Sampling-Based-Path-Planning-Library/blob/master/RRTFamilyOfPlanners.py
from .RRTFamilyOfPlanners import SamplingBasedPathPlanner

@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def __init__(self, sg: VehicleGeometry, sp: VehicleParameters):
        self.sg = sg
        self.sp = sp
        self.name: PlayerName = None
        self.goal: PlanningGoal = None
        self.lanelet_network: LaneletNetwork = None
        self.static_obstacles: Sequence[StaticObstacle] = None
        # feel free to remove/modify the following
        self.params = Pdm4arAgentParams()
        # initialization
        self.resolution = 10
        self.drawResults = True
        self.runForFullIterations = False

        self.sbpp = SamplingBasedPathPlanner()

        self.path = []
        self.Delta0 = 0
        self.next_time = 0.1
        # starting speed 
        self.Vx = 0
        self.Vy = 0
        
        self.avoidance_activation_radius = 20 # in meters, distance range from dynamic obstacles that activates the avoidance behavior
        self.extra_distance = 1.0 # extra distance to keep from obstacle, for more safety

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator at the beginning of each episode."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.static_obstacles = list(init_obs.dg_scenario.static_obstacles.values())

    def find_path(self, start, RRT_algorithm):
        
        num_iterations = int(self.area_world / 3) 
        if num_iterations < 0:
            num_iterations *= -1
        print("Initialize", RRT_algorithm, ", margin from obstacles: ", self.object_radius, " m")
        self.sbpp.RRTStar(RRT_algorithm, self.stat_obstacles, self.boundaries, start, self.goal_region, self.object_radius, self.steer_distance,
                                    None, num_iterations, self.resolution, self.runForFullIterations, self.drawResults)   

        while not self.sbpp.RRTFamilySolver.path:
            self.object_radius *= 0.9
            print("Re-initialize", RRT_algorithm, ", margin from obstacles: ", self.object_radius, " m")
            num_iterations = int(self.area_world / 3)      
            del self.sbpp
            self.sbpp = SamplingBasedPathPlanner()
            self.sbpp.RRTStar(RRT_algorithm, self.stat_obstacles, self.boundaries, start, self.goal_region, self.object_radius, self.steer_distance, 
                                    None, num_iterations, self.resolution, self.runForFullIterations, self.drawResults)   

        print("Initial path found!")

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """ This method is called by the simulator at each time step.
        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """
        # todo implement here some better planning
        # rnd_acc = random.random() * self.params.param1
        # rnd_ddelta = (random.random()-.5) * self.params.param1

        #### keys: x -- CoG x location 
        # y -- CoG y location
        # psi -- CoG heading
        # vx -- CoG longitudinal velocity
        # delta -- steering angle of the front wheel
        current_state = sim_obs.players[self.name].state
        current_time = sim_obs.time
        occupancy = sim_obs.players[self.name].occupancy

        # how to access variables
        initial_position = (current_state.x, current_state.y)

        # goal state given as a polygon where arrive
        goal = self.goal.goal.centroid
        goal_x = goal.x
        goal_y = goal.y

        #### parameters
        param = self.sp
        # actuation limits and state constraints
        min_acc, max_acc = param.acc_limits
        max_steering_rate = param.ddelta_max
        max_steering_angle = param.delta_max
        min_vel, max_vel = param.vx_limits

        #### vehicle geometry
        geometry = self.sg

        #### road network
        # create a list with center vertex
        center_vertex = []
        for element in self.lanelet_network.lanelets:
            center_vertex.append(element.center_vertices[0])
            center_vertex.append(element.center_vertices[1])

        ## x
        min_x = 0
        max_x = 0
        for element in center_vertex:
            if element[0] < min_x:
                min_x = element[0]
            if element[0] > max_x:
                max_x = element[0]

        ## y
        min_y = 0
        max_y = 0
        for element in center_vertex:
            if element[1] < min_y:
                min_y = element[1]
            if element[1] > max_y:
                max_y = element[1]

        #### map of visible obstacles
        self.stat_obstacles = [item.shape for item in self.static_obstacles]
        #### where to sample
        self.stat_obstacles.append(self.lanelet_network.lanelet_polygons)
        
        self.max_y = max_y
        self.max_x = max_x
        self.min_y = min_y
        self.min_x = min_x
        
        self.area_world = (self.max_x - self.min_x) * (self.max_y - self.min_y)
        self.boundaries = (self.min_x, self.min_y, self.max_x, self.max_y)
        self.goal_region = self.goal.goal
        self.object_radius = max(2 * geometry.w_half, geometry.lf + geometry.lr)# + self.extra_distance
        self.steer_distance = max_steering_angle

        ################# PATH PLANNING
        Start = time.time()
        if not self.path:
            self.find_path(initial_position, "Informed RRT*")
            self.path = self.sbpp.RRTFamilySolver.path
        End = time.time()
        print("\n")
        print ('Computation time:', End - Start)
        print("\n")

        ############### LANE FOLLOWER
        # control_points = Pdm4arAgent.convert_to_control_points(self, self.path)
        # lane = DgLanelet(control_points=control_points)
        # model_params = VehicleParameters(vx_limits=[min_vel, max_vel], acc_limits=[min_acc, max_acc], \
        #    delta_max=max_steering_angle, ddelta_max=max_steering_rate)
        # speed_behaviour = SpeedBehavior(self.name)
        # controller = LFAgent(lane=lane, model_params=model_params, model_geo=geometry, speed_behavior=speed_behaviour)
        # controller.my_name = self.name
        # des_acc, des_ddelta = LFAgent.get_commands(controller, sim_obs)

        ############## FIND NEAREST OBSTACLE
        obstacles_tree = STRtree([item.shape for item in self.static_obstacles])
        nearest_obstacle = obstacles_tree.nearest(Point(current_state.x, current_state.y))
        distance_obstacle = Point(current_state.x, current_state.y).distance(nearest_obstacle)
        

        ############## CONTROLLER
        path_shapely = LineString(self.path)
        distance = path_shapely.project(Point(current_state.x, current_state.y))
        reference = path_shapely.interpolate(distance + self.object_radius)
        
        v = np.array([reference.centroid.x - current_state.x, reference.centroid.y - current_state.y])
        ref_pose_orientation = np.arctan2(v[1], v[0])

        # acceleration x axis
        self.reference = reference.centroid.x
        self.measurement = current_state.x
        self.last_request_at = 0
        self.last_integral_error = 0
        self.last_proportianal_error = 0
            

        self.params = PIDParam(1.5, 1, 2, (min_vel, max_vel))
        des_v_x = PID.get_control(self, at=0)

        des_acc_x = (des_v_x - self.Vx) / (self.next_time - float(current_time))
        
        self.Vx = des_v_x

        # accelleration y axis
        self.reference = reference.centroid.y
        self.measurement = current_state.y
        self.last_request_at = 0
        self.last_integral_error = 0
        self.last_proportianal_error = 0

        self.params = PIDParam(1.5, 1, 2, (min_vel, max_vel))
        des_v_y = PID.get_control(self, at = 0)
        des_acc_y = (des_v_y - self.Vy) / (self.next_time - float(current_time))
        
        self.Vy = des_v_y

        des_acc = (des_acc_x ** 2 + des_acc_y ** 2) ** 0.5

        # if the path is straight, increase the speed
        # if des_acc == 0:# and ((goal_x - current_state.x)**2 + (goal_y - current_state.y)**2) ** 0.5 < 30:
        #    des_acc = max_acc / 2

        # steering rate
        self.reference = ref_pose_orientation
        self.measurement = current_state.psi
        self.last_request_at = 0
        self.last_integral_error = 0
        self.last_proportianal_error = 0

        self.params = PIDParam(0.75, 1, 1, (-max_steering_angle, max_steering_angle))
        des_delta = PID.get_control(self, at = 0)
        des_ddelta = (des_delta - self.Delta0) / (self.next_time - float(current_time))

        
        self.Delta0 = des_delta
        
        self.next_time += 0.1 
        
        #################### BRAKE IF NEEDED
        if distance_obstacle <= 10:
            print('Start braking')
            des_acc = min_acc
            des_ddelta = 0
            if current_state.vx <= 0:
                current_state.vx = 0
                des_acc = 0
                print('Braking completed')
        ## saturation at 8 m/s
        #if current_state.vx >= 8:
        #    current_state.vx = 8
        #    des_acc = min_acc
        
        if des_ddelta >= max_steering_rate or des_ddelta <= - max_steering_angle:
            if current_state.vx > 5:
                #current_state.vx = 5
                des_acc = min_acc

        print("\n")
        print('Agent name:', self.name)
        print("\n")
        print('Steering rate:', des_ddelta, 'Acceleration:', des_acc)
        print("\n")
        print('Current velocity:', current_state.vx)
        print("\n")
        print('Coordinates. x:', current_state.x, 'y:', current_state.y)
        print("\n")
        print('Distance 2 goal:', ((goal_x - current_state.x)**2 + (goal_y - current_state.y)**2) ** 0.5, 'Distance from obstacles:', distance_obstacle)
        print("\n")

        # return acceleration and steering rate
        return VehicleCommands(acc=des_acc, ddelta=des_ddelta)
