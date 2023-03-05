from typing import Tuple

import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy, Cell, State, Action
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
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
    def policy_evaluation(grid_mdp: GridMdp, value_func, policy):
        new_value_func = np.zeros_like(grid_mdp.grid).astype(float)
        
        gamma=grid_mdp.gamma
        grid=grid_mdp.grid
        start=np.where(grid ==Cell.START)
        # grid dimension
        rows=len(grid)
        columns=len(grid[0])

        threshold=0.001
        Delta=threshold+1.0
        
        while Delta>threshold:
            Delta=0.0
            for i in range(rows):
                for j in range(columns):
                    # find neighbour
                    neighbour=PolicyIteration.find_neighbours(i,j)
                    new_value=0
                    for next in neighbour:
                        if PolicyIteration.check(next,rows,columns):
                            # get transition probability -> from s to s'
                            T=grid_mdp.get_transition_prob([i,j],policy[i][j],next)
                            # get V(s')
                            V=value_func[next[0]][next[1]]
                            # get reward of current cell
                            R=grid_mdp.stage_reward([i,j],policy[i][j])
                            new_value+=T*(R+gamma*V)
                        else:
                            # get transition probability -> from s to s'
                            T=grid_mdp.get_transition_prob([i,j],policy[i][j],next)
                            # new robot parachute in start
                            V=value_func[start[0][0]][start[1][0]]
                            # get reward of current cell
                            R=grid_mdp.stage_reward([i,j],policy[i][j])
                            new_value+=T*(R+gamma*V)
                    # sum values for all neighbour
                    value_delta=abs(value_func[i][j]-new_value)
                    Delta=max(Delta,value_delta)
                    new_value_func[i][j]=new_value
            # update value function by taking max value for actions
            value_func=new_value_func.copy()

        return value_func

    @staticmethod
    def policy_improvement(grid_mdp:GridMdp, value_func, policy):
                
        gamma=grid_mdp.gamma
        grid=grid_mdp.grid
        start=np.where(grid ==Cell.START)
        # grid dimension
        rows=len(grid)
        columns=len(grid[0])
        change=0
        
        for i in range(rows):
            for j in range(columns):
                # return index of neighbour
                neighbour=PolicyIteration.find_neighbours(i,j)
                max_value=-1.0
                max_action=-1 
                # loop over all possible actions
                actions=PolicyIteration.poss_actions(grid,(i,j),rows,columns)
                for action in actions:
                    new_value=0
                    for next in neighbour:
                        if PolicyIteration.check(next,rows,columns):
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
                # check policy value of current cell
                if max_action!=-1 and max_action!=policy[i][j]:
                    policy[i][j]=max_action
                    change=1

        return change,policy

    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        # todo implement here
        # update
        while 1:
            value_func=PolicyIteration.policy_evaluation(grid_mdp, value_func,policy)
            change,policy=PolicyIteration.policy_improvement(grid_mdp, value_func,policy)
            if not change:
                break
        
        return value_func, policy