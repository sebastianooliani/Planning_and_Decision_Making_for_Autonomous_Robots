from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Cell, Policy, State, ValueFunc


class GridMdp:
    def __init__(self, grid: NDArray[np.int], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        # possible actions
        actions={(-1,0):Action.NORTH, (1,0):Action.SOUTH,(0,1):Action.EAST,(0,-1):Action.WEST,(0,0):Action.STAY}
        # compute position movements along x and y
        x=next_state[0]-state[0]
        y=next_state[1]-state[1]
        action_req=actions[x,y]

        cell=self.grid[state[0],state[1]]

        if cell==Cell.GOAL:
            if action==Action.STAY==action_req:
                return 1.0
            else:
                return 0.0
        elif cell==Cell.GRASS or cell==Cell.START:
            if action_req==action:
                return 0.75
            elif action_req==Action.STAY:
                return 0.0
            else:
                return 0.25/3.0
        elif cell==Cell.SWAMP:
            if action_req==action:
                return 0.5
            elif action_req==Action.STAY:
                return 0.25
            else:
                return 0.25/3.0

    def stage_reward(self, state: State, action: Action) -> float:
        # todo
        cell=self.grid[state[0],state[1]]

        if cell==Cell.GOAL:
            return 10.0
        elif cell==Cell.GRASS or cell==Cell.START:
            return -1.0
        elif cell==Cell.SWAMP:
            return -2.0


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(self,grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        pass
