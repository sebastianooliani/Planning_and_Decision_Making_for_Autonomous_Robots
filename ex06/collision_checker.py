from typing import List
from dg_commons import SE2Transform
import numpy as np
from shapely.strtree import STRtree, BaseGeometry
from shapely import geometry
from pdm4ar.exercises.ex06.collision_primitives import CollisionPrimitives
from pdm4ar.exercises_def.ex06.structures import (
    Polygon,
    GeoPrimitive,
    Point,
    Segment,
    Circle,
    Triangle,
    Path,
)

##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

COLLISION_PRIMITIVES = {
    Point: {
        Circle: lambda x, y: CollisionPrimitives.circle_point_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_point_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_point_collision(y, x),
    },
    Segment: {
        Circle: lambda x, y: CollisionPrimitives.circle_segment_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_segment_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_segment_collision_aabb(y, x),
    },
    Triangle: {
        Point: CollisionPrimitives.triangle_point_collision,
        Segment: CollisionPrimitives.triangle_segment_collision,
    },
    Circle: {
        Point: CollisionPrimitives.circle_point_collision,
        Segment: CollisionPrimitives.circle_segment_collision,
    },
    Polygon: {
        Point: CollisionPrimitives.polygon_point_collision,
        Segment: CollisionPrimitives.polygon_segment_collision_aabb,
    },
}


def check_collision(p_1: GeoPrimitive, p_2: GeoPrimitive) -> bool:
    """
    Checks collision between 2 geometric primitives
    Note that this function only uses the functions that you implemented in CollisionPrimitives class.
        Parameters:
                p_1 (GeoPrimitive): Geometric Primitive
                p_w (GeoPrimitive): Geometric Primitive
    """
    assert type(p_1) in COLLISION_PRIMITIVES, "Collision primitive does not exist."
    assert (
        type(p_2) in COLLISION_PRIMITIVES[type(p_1)]
    ), "Collision primitive does not exist."

    collision_func = COLLISION_PRIMITIVES[type(p_1)][type(p_2)]

    return collision_func(p_1, p_2)


##############################################################################################


class CollisionChecker:
    """
    This class implements the collision check ability of a simple planner for a circular differential drive robot.

    Note that check_collision could be used to check collision between given GeoPrimitives
    check_collision function uses the functions that you implemented in CollisionPrimitives class.
    """

    def __init__(self):
        pass

    def path_collision_check(self, t: Path, r: float, obstacles: List[GeoPrimitive]) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1.
            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        # list for collecting the results
        results = []

        # for each segment of the path, check for collision with obstacles
        for i in range(len(t.waypoints)-1):
            p1 = t.waypoints[i+1]
            p0 = t.waypoints[i]
            segment = Segment(p0, p1)
            for obj in obstacles:
                if isinstance(obj, (Triangle)):
                    if CollisionPrimitives.triangle_segment_collision(obj, segment):
                        results.append(i)
                        break
                    else:
                        # check distance points - robots
                        for c in [Circle(obj.v1, r), Circle(obj.v3, r), Circle(obj.v2, r)]:
                            if CollisionPrimitives.circle_segment_collision(c, segment):
                                results.append(i)
                                break
                elif isinstance(obj, (Circle)):
                    if CollisionPrimitives.circle_segment_collision(obj, segment):
                        results.append(i)
                        break
                    else:
                        # get the closest point to the circle
                        d = CollisionPrimitives.compute_distance([p0.x, p0.y], [p1.x, p1.y])
                        alpha =((p1.x - p0.x) * (obj.center.x - p0.x) + \
                            (p1.y - p0.y) * (obj.center.y - p0.y))/d**2
                        # M = closest point to the line
                        M = np.array([p0.x + (p1.x - p0.x) * alpha, \
                            p0.y + (p1.y - p0.y) * alpha])
                        centre = np.array([obj.center.x, obj.center.y])
                        if CollisionPrimitives.compute_distance(centre, M) <= obj.radius + r:
                            # check if M is in the segment
                            if CollisionPrimitives.compute_distance(M, [p0.x, p0.y]) <= d\
                                and CollisionPrimitives.compute_distance(M, [p1.x, p1.y]) <= d:
                                # circle and line intersect
                                results.append(i)
                                break
                            # if M is not in the segment
                            elif CollisionPrimitives.compute_distance(centre, [p0.x, p0.y]) <= obj.radius + r\
                                or CollisionPrimitives.compute_distance(centre, [p1.x, p1.y]) <= obj.radius + r:
                                # circle and segment intersect
                                results.append(i)
                                break
                elif isinstance(obj, (Polygon)):
                    if CollisionPrimitives.polygon_segment_collision(obj, segment):
                        results.append(i)
                        break
                    else:
                        # verify collision circle with center in vertices-segment
                        for v in obj.vertices:
                            if CollisionPrimitives.circle_segment_collision(Circle(v, r), segment):
                                results.append(i)
                                break

        return results

    def rotation_matrix(self, theta: float):
        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    def compute_robot_grid(self, t: Path, start: Point, end: Point, r: float):
        p0, p1 = t.get_boundaries()
        robot_occupancy_grid = np.zeros((2 * p1.x, 2 * p1.y), dtype=int)
        # map robot area as circle + polygon
        c1 = Circle(start, r)
        c2 = Circle(end, r)

        v = np.array([end.x - start.x, end.y - start.y])
        v1 = np.matmul(CollisionChecker.rotation_matrix(self, np.pi/2 + np.arctan2(v[1], v[0])), [r, 0, 0])
        v2 = np.matmul(CollisionChecker.rotation_matrix(self, np.pi/2 + np.arctan2(v[1], v[0])), [r, 0, 0])

        vertices_up = [start, Point(end.x + r * np.cos(np.arctan2(v[1], v[0])), end.y + r * np.sin(np.arctan2(v[1], v[0]))),\
             Point(v1[0] + start.x, v1[1] + start.y), Point(v2[0] + end.x, v2[1] + end.y)]
        poly_up = Polygon(vertices_up)
        
        # create a polygon from vertices
        v3 = np.matmul(CollisionChecker.rotation_matrix(self, np.arctan2(v[1], v[0]) - np.pi/2), [r, 0, 0])
        v4 = np.matmul(CollisionChecker.rotation_matrix(self, np.arctan2(v[1], v[0]) - np.pi/2), [r, 0, 0])
        vertices_down = [start, Point(end.x + r * np.cos(np.arctan2(v[1], v[0])), end.y + r * np.sin(np.arctan2(v[1], v[0]))),\
             Point(v3[0] + start.x, v3[1] + start.y), Point(v4[0] + end.x, v4[1] + end.y)]
        poly_down = Polygon(vertices_down)

        obstacles = [poly_up, poly_down]
        for obj in obstacles:
            a, b = obj.get_boundaries()
            if b.x < p1.x and b.y < p1.y:
                for i in range(2 * int(a.x), 2 * int(b.x)):
                    for j in range(2 * int(a.y), 2 * int(b.y)):
                        if isinstance(obj, (Polygon)):# or isinstance(obj, (Triangle)):
                            #if check_collision(obj, Point(float(i), float(j))):
                                robot_occupancy_grid[int(i)][int(j)] = 1
                        elif isinstance(obj, (Triangle)):
                            #if check_collision(obj, Point(float(i), float(j))):
                                robot_occupancy_grid[int(i)][int(j)] = 1
                        elif isinstance(obj, (Circle)):
                            if CollisionPrimitives.compute_distance([2 * obj.center.x, 2 * obj.center.y], [i,j]) <= 2 * obj.radius:
                                robot_occupancy_grid[int(i)][int(j)] = 1

        return robot_occupancy_grid

    def path_collision_check_occupancy_grid(self, t: Path, r: float, obstacles: List[GeoPrimitive]) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will generate an occupancy grid of the given map.
        Then, occupancy grid will be used to check collisions.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        # points at limit of grids
        p0, p1 = t.get_boundaries()
        # create grid of binary values
        occupancy_grid = np.zeros((2 *(p1.x - p0.x), 2 * (p1.y - p0.y)), dtype=int)
        # fill the grid

        for obj in obstacles:
            a, b = obj.get_boundaries()
            if b.x < p1.x-1 and b.y < p1.y-1:
                for i in range(2 * int(a.x), 2 * int(b.x)):
                    for j in range(2 * int(a.y), 2 * int(b.y)):
                        if isinstance(obj, (Polygon)):
                            #if CollisionPrimitives.polygon_point_collision(obj, Point(float(i), float(j))):
                                occupancy_grid[int(i)][int(j)] = 1
                        elif isinstance(obj, (Triangle)):
                            #if CollisionPrimitives.triangle_point_collision(obj, Point(float(i), float(j))):
                                occupancy_grid[int(i)][int(j)] = 1
                        elif isinstance(obj, (Circle)):
                            if CollisionPrimitives.compute_distance([2 * obj.center.x, 2 * obj.center.y], [i,j]) <= 2 * obj.radius:
                                occupancy_grid[int(i)][int(j)] = 1

        # list with segment in collision
        results = []
        
        for i in range(len(t.waypoints)-1):
            p5 = t.waypoints[i+1]
            p4 = t.waypoints[i]
            robot_occupancy_grid = CollisionChecker.compute_robot_grid(self, t, p4, p5, r)
            # dimension of robot grid
            rows = len(robot_occupancy_grid)
            col = len(robot_occupancy_grid[0])
            for j in range(rows):
                for k in range(col):
                    if robot_occupancy_grid[j][k] == 1 and occupancy_grid[j][k] == 1:
                        if i not in results:
                            results.append(i)
                        break
        
        # idea create grid to identify where robot will be and overlap grids value
        
        return results

    def path_collision_check_r_tree(self, t: Path, r: float, obstacles: List[GeoPrimitive]) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will build an R-Tree of the given obstacles.
        You are free to implement your own R-Tree or you could use STRTree of shapely module.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        geom_obstacles = []
        index = {}
        for obj in obstacles:
            if isinstance(obj, (Polygon)):
                vertices = []
                for element in obj.vertices:
                    vertices.append((element.x, element.y))
                new = geometry.Polygon(vertices)
                geom_obstacles.append(new)
                index[id(new)] = obj
            elif isinstance(obj, (Triangle)):
                vertices = ((obj.v1.x, obj.v1.y), (obj.v2.x, obj.v2.y), (obj.v3.x, obj.v3.y))
                new = geometry.Polygon(vertices)
                geom_obstacles.append(new)
                index[id(new)] = obj
            elif isinstance(obj, (Circle)):
                p = obj.center
                radius = obj.radius
                new = geometry.Point([p.x, p.y]).buffer(radius)
                geom_obstacles.append(new)
                index[id(new)] = obj

        tree = STRtree(geom_obstacles)

        results = []
        
        for i in range(len(t.waypoints)-1):
            p1 = t.waypoints[i+1]
            p0 = t.waypoints[i]
            theta = np.arctan2(p1.y - p0.y, p1.x - p0.x)
            vertix_left_start = (p0.x + r * np.cos(theta + np.pi/2), p0.y + r * np.sin(theta + np.pi/2))
            vertix_left_end = (p1.x + r * np.cos(theta + np.pi/2), p1.y + r * np.sin(theta + np.pi/2))
            vertix_right_start = (p0.x + r * np.cos(theta - np.pi/2), p0.y + r * np.sin(theta - np.pi/2))
            vertix_right_end = (p1.x + r * np.cos(theta - np.pi/2), p1.y + r * np.sin(theta - np.pi/2))
            
            segment_right = geometry.LineString([vertix_right_start, vertix_right_end])
            segment_left = geometry.LineString([vertix_left_start, vertix_left_end])
            intersection_left = tree.query(segment_left)
            intersection_right = tree.query(segment_right)

            if len(intersection_left) > 0:
                p_start = Point(vertix_left_start[0], vertix_left_start[1])
                p_end = Point(vertix_left_end[0], vertix_left_end[1])
                segment = Segment(p_start, p_end)

                for element in intersection_left:
                    actual_obstacle = index[id(element)]
                    if isinstance(actual_obstacle, (Polygon)):
                        if CollisionPrimitives.polygon_segment_collision(actual_obstacle, segment):
                            if i not in results:
                                results.append(i)
                    elif isinstance(actual_obstacle, (Triangle)):
                        if CollisionPrimitives.triangle_segment_collision(actual_obstacle, segment):
                            if i not in results:
                                results.append(i)
                    elif isinstance(actual_obstacle, (Circle)):
                        if CollisionPrimitives.circle_segment_collision(actual_obstacle, segment):
                            if i not in results:
                                results.append(i)

            if len(intersection_right) > 0:
                p_start = Point(vertix_right_start[0], vertix_right_start[1])
                p_end = Point(vertix_right_end[0], vertix_right_end[1])
                segment = Segment(p_start, p_end)

                for element in intersection_right:
                    actual_obstacle = index[id(element)] 
                    if isinstance(actual_obstacle, (Polygon)):
                        if CollisionPrimitives.polygon_segment_collision(actual_obstacle, segment):
                            if i not in results:
                                results.append(i)
                    elif isinstance(actual_obstacle, (Triangle)):
                        if CollisionPrimitives.triangle_segment_collision(actual_obstacle, segment):
                            if i not in results:
                                results.append(i)
                    elif isinstance(actual_obstacle, (Circle)):
                        if CollisionPrimitives.circle_segment_collision(actual_obstacle, segment):
                            if i not in results:
                                results.append(i)
            
            intersection_start = tree.query(geometry.Point([p0.x, p0.y]).buffer(r))
            intersection_end = tree.query(geometry.Point([p1.x, p1.y]).buffer(r))

            if len(intersection_start) > 0:
                for element in intersection_start:
                    actual_obstacle = index[id(element)] 
                    if isinstance(actual_obstacle, (Polygon)):
                        for i in range(len(actual_obstacle.vertices)-1):
                            segment = Segment(actual_obstacle.vertices[i], actual_obstacle.vertices[i+1])
                            if CollisionPrimitives.circle_segment_collision(Circle(p0, r), segment):
                                if i not in results:
                                    results.append(i)
                    if isinstance(actual_obstacle, (Triangle)):
                        for s in [Segment(actual_obstacle.v1, actual_obstacle.v2), Segment(actual_obstacle.v2, actual_obstacle.v3), Segment(actual_obstacle.v3, actual_obstacle.v1)]:
                            if CollisionPrimitives.circle_segment_collision(Circle(p0, r), s):
                                if i not in results:
                                    results.append(i)
                    if isinstance(actual_obstacle, (Circle)):
                        if CollisionPrimitives.compute_distance([p0.x, p0.y], [actual_obstacle.center.x, actual_obstacle.center.y]) <= r + actual_obstacle.radius:
                            if i not in results:
                                    results.append(i)

            if len(intersection_end) > 0:
                for element in intersection_end:
                    actual_obstacle = index[id(element)] 
                    if isinstance(actual_obstacle, (Polygon)):
                        for i in range(len(actual_obstacle.vertices)-1):
                            segment = Segment(actual_obstacle.vertices[i], actual_obstacle.vertices[i+1])
                            if CollisionPrimitives.circle_segment_collision(Circle(p1, r), segment):
                                if i not in results:
                                    results.append(i)
                    if isinstance(actual_obstacle, (Triangle)):
                        for s in [Segment(actual_obstacle.v1, actual_obstacle.v2), Segment(actual_obstacle.v2, actual_obstacle.v3), Segment(actual_obstacle.v3, actual_obstacle.v1)]:
                            if CollisionPrimitives.circle_segment_collision(Circle(p1, r), s):
                                if i not in results:
                                    results.append(i)
                    if isinstance(actual_obstacle, (Circle)):
                        if CollisionPrimitives.compute_distance([p1.x, p1.y], [actual_obstacle.center.x, actual_obstacle.center.y]) <= r + actual_obstacle.radius:
                            if i not in results:
                                    results.append(i)
        
        return results

    def compute_transformation_matrix(self, pose: SE2Transform):
        return np.array([[np.cos(pose.theta), -np.sin(pose.theta), 0, pose.p[0]],\
            [np.sin(pose.theta), np.cos(pose.theta), 0, pose.p[1]], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: SE2Transform,
        next_pose: SE2Transform,
        observed_obstacles: List[GeoPrimitive],
    ) -> bool:
        """
        Returns there exists a collision or not during the movement of a circular differential drive robot until its next pose.

            Parameters:
                    r (float): Radius of circular differential drive robot
                    current_pose (SE2Transform): Current pose of the circular differential drive robot
                    next_pose (SE2Transform): Next pose of the circular differential drive robot
                    observed_obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives in robot frame
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        # both wrt world frame
        T0 = CollisionChecker.compute_transformation_matrix(self, current_pose)
        T1 = CollisionChecker.compute_transformation_matrix(self, next_pose)
        # transformation from current to next pose
        T_mov = np.dot(np.linalg.inv(T0),T1)
        # last column give the translation vector from current to next pose
        translation1 = np.array([T_mov[0][3], T_mov[1][3]])
        # next pose wrt robot frame 
        points = [Point(0, 0), Point(translation1[0], translation1[1])]
        path = Path(points)

        if len(CollisionChecker.path_collision_check(self, path, r, observed_obstacles)) != 0:
            return True
        return False

    def safety_certificate_check(self, p0: Point, theta: float, d_obs: float, element):
        tan = np.array([p0.x + d_obs * np.cos(theta), p0.y + d_obs * np.sin(theta)])
        new_d_obs = element.distance(geometry.Point([tan[0], tan[1]]))
        return new_d_obs

    def path_collision_check_safety_certificate(self, t: Path, r: float, obstacles: List[GeoPrimitive]) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will implement the safety certificates procedure for collision checking.
        You are free to use shapely to calculate distance between a point and a GoePrimitive.
        For more information, please check Algorithm 1 inside the following paper:
        https://journals.sagepub.com/doi/full/10.1177/0278364915625345.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        # define S_free and S_obs
        results = []
        # compute distance point - obstacle with shapely
        geom_obstacles = []

        for obj in obstacles:
                if isinstance(obj, (Polygon)):
                    vertices = []
                    for element in obj.vertices:
                        vertices.append((element.x, element.y))
                    new = geometry.Polygon(vertices)
                    geom_obstacles.append(new)
                if isinstance(obj, (Triangle)):
                    vertices = ((obj.v1.x, obj.v1.y), (obj.v2.x, obj.v2.y), (obj.v3.x, obj.v3.y))
                    new = geometry.Polygon(vertices)
                    geom_obstacles.append(new)
                if isinstance(obj, (Circle)):
                    p = obj.center
                    radius = obj.radius
                    new = geometry.Point([p.x, p.y]).buffer(radius)
                    geom_obstacles.append(new)

        for i in range(len(t.waypoints)-1):
            p1 = t.waypoints[i+1]
            p0 = t.waypoints[i]
            theta = np.arctan2(p1.y - p0.y, p1.x - p0.x)

            length = CollisionPrimitives.compute_distance([p0.x, p0.y], [p1.x, p1.y])
            for element in geom_obstacles:
                # nearest obstacle
                d_obs = element.distance(geometry.Point([p0.x, p0.y]))
                if d_obs <= r:
                    # collision
                    if i not in results:
                        results.append(i)
                while length + r >= d_obs:
                    new_d_obs = CollisionChecker.safety_certificate_check(self, p0, theta, d_obs, element)
                    # what safety_certificate_check does
                    # tan = np.array([p0.x + d_obs * np.cos(theta), p0.y + d_obs * np.sin(theta)])
                    # new_d_obs = element.distance(geometry.Point([tan[0], tan[1]]))
                    if new_d_obs <= r:
                        # collision
                        if i not in results:
                            results.append(i)
                        break
                    d_obs = d_obs + new_d_obs
        
        return results
