from abc import abstractmethod, ABC
from typing import Tuple

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        # todo implement here your solution
        # visited
        V=[start]
        # initialize queue w/ starting state
        Q=[(start,[start])]
        # initialize visited state w/starting state
        OpenedNodes=[start]
        while Q:
            # take the first element of Q
            (current_node,Path)=Q.pop(0)
            # if we have found a path
            if current_node == goal:
                if current_node not in OpenedNodes:
                    # insert node in V
                    OpenedNodes.append(current_node)
                return Path,OpenedNodes
            # for all the following reachable nodes
            if current_node not in OpenedNodes:
                # insert node in V
                V.append(current_node)
                # if current_node not in Q:
                OpenedNodes.append(current_node)
            sort=sorted(graph[current_node])
            add=[]
            for neighbour in sort:
                # if the node has not been visited
                if neighbour not in V:
                    V.append(neighbour)
                    # insert node in Q at the front
                    add.append((neighbour,Path+[neighbour]))
            Q=add+Q
        # if no paths are found                            
        return [], OpenedNodes


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        # todo implement here your solution
        V=[start]
        # initialize queue w/ starting state
        Q=[(start,[start])]
        # initialize visited state w/starting state
        OpenedNodes=[start]
        while Q:
            # take the first element of Q
            (current_node,Path)=Q.pop(0)
            # if we have found a path
            if current_node == goal:
                if current_node not in OpenedNodes:
                    # insert node in V
                    OpenedNodes.append(current_node)
                return Path,OpenedNodes
            if current_node not in OpenedNodes:
                # insert node in V
                V.append(current_node)
                # if current_node not in Q:
                OpenedNodes.append(current_node)
            # for all the following reachable nodes
            sort=sorted(graph[current_node])
            for neighbour in sort:
                # if the node has not been visited
                if V.count(neighbour)==0:
                    # insert node in Q at the front
                    Q.append((neighbour,Path+[neighbour]))
                    # insert node in V
                    V.append(neighbour)
                    # OpenedNodes.append(neighbour)
        # if no paths are found 
        return [], OpenedNodes


class IterativeDeepening(GraphSearch):
    def depth_first_search(self, graph: AdjacencyList, start: X, goal: X, depth) -> Tuple[Path, OpenedNodes]:
        # todo implement here your solution
        V=[start]
        # initialize depth level
        Depth={start: 1}
        # initialize queue w/ starting state
        Q=[(start,[start])]
        # initialize visited state w/starting state
        OpenedNodes=[]
        current_node=start
        while Q and Depth[current_node]<=depth:
            # take the first element of Q
            (current_node,Path)=Q.pop(0)
            OpenedNodes.append(current_node)
            # if we have found a path
            if current_node == goal:
                if current_node not in OpenedNodes:
                    # insert node in V
                    OpenedNodes.append(current_node)
                return Path,OpenedNodes
            # for all the following reachable nodes
            sort=sorted(graph[current_node])
            add=[]
            for neighbour in sort:
                # if the node has not been visited
                if neighbour not in V:
                    V.append(neighbour)
                    # insert node in Q at the front
                    Depth[neighbour] = Depth[current_node] + 1
                    if Depth[neighbour] <= depth:
                        add.append((neighbour,Path+[neighbour]))
            Q=add+Q
        # if no paths are found
        return [], OpenedNodes
    
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        
        max_depth = len(graph)
        depth = 1

        while depth <= max_depth:
            Path, OpenedNodes = self.depth_first_search(graph, start, goal, depth)
            if Path:
                return Path, OpenedNodes
            depth = depth + 1

        return [], OpenedNodes
