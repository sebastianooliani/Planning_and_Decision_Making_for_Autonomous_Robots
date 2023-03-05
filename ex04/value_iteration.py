from typing import Tuple

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc, Cell, State, Action
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    def find_neighbours(x:int, y:int):
    # check in N,S,W,E directions + stay
        return [(x,y),(x+1,y),(x,y+1),(x-1,y),(x,y-1)]
    
    @staticmethod
    def poss_actions(grid, x:State, row:int, column:int):
        cell=grid[x[0]][x[1]]
        actions=[Action.NORTH,Action.WEST,Action.SOUTH,Action.EAST]

        if cell==Cell.GRASS or cell==Cell.SWAMP or Cell==Cell.START:
            if x[0]==0:
                actions.remove(Action.NORTH)
            elif x[0]==row-1:
                actions.remove(Action.SOUTH)
            if x[1]==0:
                actions.remove(Action.WEST)
            elif x[1]== column-1:
                actions.remove(Action.EAST)
            return actions
        else:
            actions.append(Action.STAY)
            return actions
    
    @staticmethod
    def check(x:State,row,column):
        if x[0]<0 or x[0]>row-1:
            return 0
        if x[1]<0 or x[1]>column-1:
            return 0
        return 1
    
    @staticmethod
    def value_iteration(grid_mdp: GridMdp, value_func, policy):
        new_value_func = np.zeros_like(grid_mdp.grid).astype(float)
        
        gamma=grid_mdp.gamma
        grid=grid_mdp.grid
        start=np.where(grid ==Cell.START)
        # start is a matrix
        # grid dimension
        rows=len(grid)
        columns=len(grid[0])

        # define improvement
        Delta=0.0
        
        for i in range(rows):
            for j in range(columns):
                # find neighbour
                neighbour=ValueIteration.find_neighbours(i,j)
                # find max value of V[i][j] -> worst case
                max_value=-1.0
                # find action that maximize V. Since all actions are positive:
                max_action=-1 
                # loop over all possible actions
                actions=ValueIteration.poss_actions(grid,(i,j),rows,columns)
                for action in actions:
                    new_value=0
                    # check neighbourhood
                    for next in neighbour:
                        # if next is in the map range
                        if ValueIteration.check(next,rows,columns):
                            # get transition probability -> from s to s'
                            T=grid_mdp.get_transition_prob([i,j],action,next)
                            # get V(s')
                            V=value_func[next[0]][next[1]]
                            # get reward of current cell
                            R=grid_mdp.stage_reward([i,j],action)
                            new_value+=T*(R+gamma*V)
                        else:
                            # get transition probability -> from s to s'
                            T=grid_mdp.get_transition_prob([i,j],action,next)
                            # new robot parachute in start
                            V=value_func[start[0][0]][start[1][0]]
                            # get reward of current cell
                            R=grid_mdp.stage_reward([i,j],action)
                            new_value+=T*(R+gamma*V)
                    # for every action, choose the one that gives max value
                    if max_action==-1 or max_value<new_value:
                        max_value=new_value
                        max_action=action

                # compute difference from value of current cell and max value
                value_delta=abs(value_func[i][j]-max_value)
                # take max difference value
                Delta=max(Delta,value_delta)
                # update value function and policy by taking max value for actions
                new_value_func[i][j]=max_value
                policy[i][j]=max_action

        return Delta, new_value_func, policy

    @staticmethod
    def compute_policy(grid_mdp:GridMdp, value_func, policy):      
        gamma=grid_mdp.gamma
        grid=grid_mdp.grid
        start=np.where(grid ==Cell.START)
        # grid dimension
        rows=len(grid)
        columns=len(grid[0])
        
        for i in range(rows):
            for j in range(columns):
                # find neighbour
                neighbour=ValueIteration.find_neighbours(i,j)
                # find max value of V[i][j]
                max_value=-1.0
                # find action that maximize V
                max_action=-1 
                # loop over all possible actions
                actions=ValueIteration.poss_actions(grid,(i,j),rows,columns)
                for action in actions:
                    new_value=0
                    # check neighbourhood
                    for next in neighbour:
                        if ValueIteration.check(next,rows,columns):
                            # get transition probability -> from s to s'
                            T=grid_mdp.get_transition_prob([i,j],action,next)
                            # get V(s')
                            V=value_func[next[0]][next[1]]
                            # get reward of current cell
                            R=grid_mdp.stage_reward([i,j],action)
                            new_value+=T*(R+gamma*V)
                        else:
                            # robot is lost, get transition probability -> from s to s'
                            T=grid_mdp.get_transition_prob([i,j],action,next)
                            # new robot parachute in start
                            V=value_func[start[0][0]][start[1][0]]
                            # get reward of current cell
                            R=grid_mdp.stage_reward([i,j],action)
                            new_value+=T*(R+gamma*V)
                    # for every action, choose the one that gives max value
                    if max_action==-1 or max_value<new_value:
                        max_value=new_value
                        max_action=action

                if max_action!=-1:
                # update policy with max action in current cell
                    policy[i][j]=max_action

        return policy

    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        # todo implement here
        threshold=0.001
        delta, value_func, policy = ValueIteration.value_iteration(grid_mdp, value_func,policy)   
        # update until there is an improvement in value function
        while delta>threshold:
            delta, value_func, policy = ValueIteration.value_iteration(grid_mdp, value_func,policy)
        # compute policy for final value function
        policy=ValueIteration.compute_policy(grid_mdp, value_func,policy)

        return value_func, policy