import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from pdm4ar.exercises_def.ex07.structures import (
    ProblemVoyage,
    OptimizationCost,
    Island,
    Constraints,
    Feasibility,
    SolutionVoyage,
)

def create_A(islands, constraints: Constraints, cost: OptimizationCost):
    number_of_island = len(islands) 
    length = islands[number_of_island-1].arch

    Order = np.zeros((length + 1, number_of_island))
    for i in range(len(Order)):
        for j in range(len(Order[0])):
            if j == islands[j].id and i == islands[j].arch:
                Order[i][j] = 1
    
    
    if constraints.min_nights_individual_island is not None:
        A = np.zeros((length + 1, number_of_island))
        for i in range(len(A)):
            for j in range(len(A[0])):
                if j == islands[j].id and i == islands[j].arch and \
                    islands[j].nights >= constraints.min_nights_individual_island:
                    A[i][j] = 1
                if (i == 0 or i == length) and (j == 0 or j == number_of_island-1):
                    A[i][j] = 1
                    break
                
        Order = np.block([[Order], [A]])
    
    
    if constraints.min_total_crew is not None:
        B = np.zeros((length, number_of_island))
        for i in range(len(B)):
            for j in range(len(B[0])):
                if islands[j].arch <= i+1:
                    B[i][j] = islands[j].delta_crew

        Order = np.block([[Order], [B]])


    if constraints.max_total_crew is not None:
        C = np.zeros((length, number_of_island))
        for i in range(len(C)):
            for j in range(len(C[0])):
                if islands[j].arch <= i+1:
                    C[i][j] = islands[j].delta_crew

        Order = np.block([[Order], [C]])

    
    if constraints.max_duration_individual_journey is not None:
        D = np.zeros((length, number_of_island))
        for i in range(len(D)):
            for j in range(len(D[0])):
                if islands[j].arch <= i+1:
                    if islands[j].arch == i:
                        D[i][j] = - islands[j].departure
                    elif islands[j].arch == i + 1:
                        D[i][j] = islands[j].arrival

        Order = np.block([[Order], [D]])

    if constraints.max_L1_distance_individual_journey is not None:
        E = np.zeros((4 * length, number_of_island))
        for i in range(len(E)):
            for j in range(len(E[0])):
                if islands[j].arch <= i+1:
                    if (islands[j].arch * 4 == i) and i % 2 == 0:
                        E[i][j] = - (islands[j].x + islands[j].y)
                        E[i + 1][j] = - (islands[j].x - islands[j].y)
                        E[i + 2][j] = + (islands[j].x + islands[j].y)
                        E[i + 3][j] = + (islands[j].x - islands[j].y)
                    elif (islands[j].arch * 4 - 4 == i) and i % 2 == 0:
                        E[i][j] = + (islands[j].x + islands[j].y)
                        E[i + 1][j] = + (islands[j].x - islands[j].y)
                        E[i + 2][j] = - (islands[j].x + islands[j].y)
                        E[i + 3][j] = - (islands[j].x - islands[j].y)

        Order = np.block([[Order], [E]])

    if cost == OptimizationCost.min_max_sailing_time:
        F = np.zeros((length, number_of_island))
        for i in range(len(F)):
            for j in range(len(F[0])):
                if islands[j].arch <= i+1:
                    if islands[j].arch == i:
                        F[i][j] = - islands[j].departure
                    elif islands[j].arch == i + 1:
                        F[i][j] = islands[j].arrival

        a = np.zeros((len(Order), 1))
        b = - np.ones((length, 1))

        Order = np.block([[Order, a], [F, b]])

    if cost == OptimizationCost.min_total_travelled_L1_distance:
        G = np.zeros((4 * length, number_of_island))
        for i in range(len(G)):
            for j in range(len(G[0])):
                if islands[j].arch <= i+1:
                    if (islands[j].arch * 4 == i) and i % 2 == 0:
                        G[i][j] = - (islands[j].x + islands[j].y)
                        G[i + 1][j] = - (islands[j].x - islands[j].y)
                        G[i + 2][j] = + (islands[j].x + islands[j].y)
                        G[i + 3][j] = + (islands[j].x - islands[j].y)
                    elif (islands[j].arch * 4 - 4 == i) and i % 2 == 0:
                        G[i][j] = + (islands[j].x + islands[j].y)
                        G[i + 1][j] = + (islands[j].x - islands[j].y)
                        G[i + 2][j] = - (islands[j].x + islands[j].y)
                        G[i + 3][j] = - (islands[j].x - islands[j].y)

        a = np.zeros((len(Order), length))
        b = np.zeros((len(G), length))

        for i in range(len(b)):
            for j in range(len(b[0])):
                if i == 0 and j == 0:
                    b[i][j] = - 1
                    b[i + 1][j] = - 1
                    b[i + 2][j] = - 1
                    b[i + 3][j] = - 1
                elif j != 0:
                    if i / j == 4:
                        b[i][j] = - 1
                        b[i + 1][j] = - 1
                        b[i + 2][j] = - 1
                        b[i + 3][j] = - 1

        Order = np.block([[Order, a], [G, b]])

    return Order

def create_b(islands, constraints: Constraints, problem):
    number_of_island = len(islands) 
    length = islands[number_of_island-1].arch

    # voyage order
    b_l = np.ones(length + 1)
    b_u = np.ones(length + 1)

    if constraints.min_nights_individual_island is not None:
        u = np.zeros(length + 1)
        l = np.ones(length + 1)
        u[:] = np.inf
        b_l = np.block([b_l, l])
        b_u = np.block([b_u, u])
    
    if constraints.min_total_crew is not None:
        l = np.ones(length) * (constraints.min_total_crew - problem.start_crew)
        u = np.zeros(length)
        u[:] = np.inf
        b_l = np.block([b_l, l])
        b_u = np.block([b_u, u])
        
    if constraints.max_total_crew is not None:
        u = np.ones(length) * (constraints.max_total_crew - problem.start_crew)
        l = np.zeros(length)
        l[:] = -np.inf
        b_l = np.block([b_l, l])
        b_u = np.block([b_u, u])

    if constraints.max_duration_individual_journey is not None:
        u = np.ones(length) * constraints.max_duration_individual_journey
        l = np.zeros(length)
        l[:] = - np.inf
        b_l = np.block([b_l, l])
        b_u = np.block([b_u, u])

    if constraints.max_L1_distance_individual_journey is not None:
        u = np.ones(4 * length) * constraints.max_L1_distance_individual_journey
        l = np.zeros(4 * length)
        l[:] = - np.inf
        b_l = np.block([b_l, l])
        b_u = np.block([b_u, u])

    if problem.optimization_cost == OptimizationCost.min_max_sailing_time:
        l = - np.inf * np.ones(length)
        u = np.zeros(length)
        b_l = np.block([b_l, l])
        b_u = np.block([b_u, u])

    if problem.optimization_cost == OptimizationCost.min_total_travelled_L1_distance:
        l = - np.inf * np.ones(4 * length)
        u = np.zeros(4 * length)
        b_l = np.block([b_l, l])
        b_u = np.block([b_u, u])

    return b_l, b_u

def create_c(islands, cost, start_crew):
    number_of_island = len(islands)
    length = islands[number_of_island-1].arch
    c = np.zeros(number_of_island)

    if cost == OptimizationCost.min_total_nights:
        for i in range(number_of_island):
            c[i] = islands[i].nights

    elif cost == OptimizationCost.max_final_crew:
        c[0] = - start_crew - islands[0].delta_crew
        for i in range(1, number_of_island):
            c[i] = - islands[i].delta_crew

    elif cost == OptimizationCost.min_total_sailing_time:
        for i in range(number_of_island):
            c[i] = (islands[i].arrival - islands[i].departure)

    elif cost == OptimizationCost.min_total_travelled_L1_distance:
        c1 = np.zeros(number_of_island)
        c2 = np.ones(length)
        c = np.block([c1, c2])

    elif cost == OptimizationCost.min_max_sailing_time:
        c = np.zeros(number_of_island + 1)
        c[-1] = 1

    return c

def create_voyage(res, islands):
    voyage = []
    length = len(islands)

    for i in range(length):
        if np.round(res.x[i]) == 1:
            voyage.append(i)

    return voyage


def solve_optimization(problem: ProblemVoyage) -> SolutionVoyage:
    """
    Solve the optimization problem enforcing the requested constraints.

    Parameters
    ---
    problem : ProblemVoyage
        Contains the problem data: cost to optimize, starting crew, tuple of islands,
        and information about the requested constraint (the constraints not set to `None` +
        the `voyage_order` constraint)

    Returns
    ---
    out : SolutionVoyage
        Contains the feasibility status of the problem, and the optimal voyage plan
        as a list of ints if problem is feasible, else `None`.
    """
    ##### CONSTRAINTS MATRICES #####
    # matrix A
    A = create_A(problem.islands, problem.constraints, problem.optimization_cost)

    # vectors b
    b_l, b_u = create_b(problem.islands, problem.constraints, problem)

    ##### COST VECTOR #####
    c = create_c(problem.islands, problem.optimization_cost, problem.start_crew)
    
    constraint = LinearConstraint(A, b_l, b_u)
    
    ##### BOUNDS DECISION VARIABLES #####
    # build bounds vector
    bounds = Bounds(0, 1)

    integrality = np.ones_like(c)
    if problem.optimization_cost == OptimizationCost.min_max_sailing_time:
        integrality[-1] = 0
        new_bound = np.ones_like(c)
        new_bound[-1] = np.inf
        bounds = Bounds(0, new_bound)
    if problem.optimization_cost == OptimizationCost.min_total_travelled_L1_distance:
        integrality = np.zeros_like(c)
        integrality[0 : len(problem.islands)] = 1
        new_bound = np.ones_like(c) * np.inf
        new_bound[0 : len(problem.islands)] = 1
        bounds = Bounds(0, new_bound)


    res = milp(c = c, integrality = integrality, constraints = constraint, bounds = bounds)

    voyage_plan = None
    if res.success:
        voyage_plan = create_voyage(res, problem.islands)
        feasibility = Feasibility.feasible
    else:
        feasibility = Feasibility.unfeasible

    return SolutionVoyage(feasibility, voyage_plan)
