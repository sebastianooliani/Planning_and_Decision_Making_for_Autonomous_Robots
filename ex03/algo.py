from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sqrt
from osmnx.distance import great_circle_vec

from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # need to introduce weights!
        pass

@dataclass
class UniformCostSearch(InformedGraphSearch):
    def determine_path(self, state: X, Parents: dict) -> Path:
        Path = [state]
        previous = Parents[state]
        while previous is not None:
            Path.insert(0, previous)
            previous = Parents[previous]
        return Path
    
    def path(self, start: X, goal: X) -> Path:
        # todo
        # cost to reach initial state
        costToReach={start: 0}
        # cost to reach of element in queue
        q_costToReach={start: 0}
        # initialize the queue
        Q=[start]
        V=[start]
        # parents
        Parents={start: None}
        while Q:
            # select node with lowest cost in Q
            key=list(q_costToReach.keys())
            low=list(q_costToReach.values())
            lowest=min(low)
            node = Q.pop(Q.index(key[low.index(lowest)]))
            q_costToReach.pop(node)
            if node == goal:
                Path = self.determine_path(node, Parents)
                return Path
            # neighbour nodes
            neighbour=self.graph.adj_list[node]
            for state in neighbour:
                newCostToReach=costToReach[node]+self.graph.get_weight(node,state)
                if state not in V:
                    # insert if not visited
                    Q.append(state)
                    V.append(state)
                    costToReach[state]=newCostToReach
                    q_costToReach[state]=newCostToReach
                    Parents[state]=node
                elif newCostToReach < costToReach[state]:
                    # update
                    costToReach[state]=newCostToReach
                    q_costToReach[state]=newCostToReach
                    Parents[state]=node
        # no path found
        return []

@dataclass
class Astar(InformedGraphSearch):
    def determine_path(self, state, Parents):
        Path = [state]
        previous = Parents[state]
        while previous is not None:
            Path.insert(0, previous)
            previous = Parents[previous]
        return Path

    def heuristic(self, u: X, v: X) -> float:
        # todo
        # admissible heuristic requires that h<=h*
        # for all states
        # for example we can choose euclidean distance
        x1,y1=self.graph.get_node_coordinates(u)
        x2,y2=self.graph.get_node_coordinates(v)
        distance=sqrt((x2-x1)**2+(y2-y1)**2)
        speed=TravelSpeed.CITY.value
        time=distance/speed
        return time
        
    def path(self, start: X, goal: X) -> Path:
        # todo
        # cost to reach initial state
        costToReach={start: 0}
        # cost to reach of element in queue
        q_costToReach={start: 0}
        # initialize the queue
        Q=[start]
        V=[start]
        # parents
        Parents={start: None}
        while Q:
            # select node with lowest cost in Q
            key=list(q_costToReach.keys())
            low=list(q_costToReach.values())
            lowest=min(low)
            node = Q.pop(Q.index(key[low.index(lowest)]))
            q_costToReach.pop(node)
            if node == goal:
                Path = self.determine_path(node, Parents)
                return Path
            # neighbour nodes
            neighbour=self.graph.adj_list[node]
            for state in neighbour:
                newCostToReach=costToReach[node]+self.graph.get_weight(node,state)
                if state not in V:
                    #insert if not visited
                    Q.append(state)
                    V.append(state)
                    costToReach[state]=newCostToReach
                    q_costToReach[state]=newCostToReach+self.heuristic(node,state)
                    Parents[state]=node
                elif newCostToReach < costToReach[state]:
                    # update
                    costToReach[state]=newCostToReach
                    q_costToReach[state]=newCostToReach+self.heuristic(node,state)
                    Parents[state]=node
        # no path found
        return []



def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
