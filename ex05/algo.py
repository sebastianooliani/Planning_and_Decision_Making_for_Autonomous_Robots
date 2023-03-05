from typing import Sequence

from dg_commons import SE2Transform

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[SE2Transform]:
        """ Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """ Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow 
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # TODO implement here your solution
    return DubinsParam(min_radius=wheel_base/np.tan(max_steering_angle))


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    # pos = np.array(current_config.p)
    theta = current_config.theta

    center_left = SE2Transform([current_config.p[0] - radius * np.sin(theta), current_config.p[1] + radius * np.cos(theta)], 0)
    center_right = SE2Transform([current_config.p[0] + radius * np.sin(theta) , current_config.p[1] - radius * np.cos(theta)], 0)
    
    left_circle = Curve.create_circle(center=center_left, config_on_circle=current_config, radius=radius, curve_type=DubinsSegmentType.LEFT)
    right_circle = Curve.create_circle(center=center_right, config_on_circle=current_config, radius=radius, curve_type=DubinsSegmentType.RIGHT)
    
    return TurningCircle(left_circle, right_circle)

def distance_between_points(start, end) -> float:
    return ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5

def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> List[Line]:
    # TODO implement here your solution
    if circle_start.gear == Gear.FORWARD and circle_end.gear == Gear.FORWARD:
        # centers of circles
        center_start = circle_start.center
        center_end = circle_end.center
        # coordinate of centers
        pos_start = np.array(center_start.p)
        pos_end = np.array(center_end.p)
        # radii
        r_start = circle_start.radius
        r_end = circle_end.radius
        # direction of rotation
        dir1 = circle_start.type
        dir2 = circle_end.type

        D = distance_between_points(pos_start,pos_end)

        # coincidents circles -> no tangent
        if pos_start[0] == pos_end[0] and pos_start[1] == pos_end[1]:
            return []

        # inner tangent
        if dir1 != dir2:
            r = r_start + r_end
            if (abs(r))/D > 1 or (abs(r))/D < -1:
                return []
            phi=np.arccos((abs(r))/(D))
            n = 1
        # outer tangent
        else:
            r = r_start - r_end
            if (abs(r))/D > 1 or (abs(r))/D < -1:
                return []
            phi=np.arccos((abs(r))/(D))
            n = 0
        
        v1 = pos_end - pos_start
        
        if dir1 == DubinsSegmentType.LEFT:
            phi *= -1
        
        theta = phi + np.arctan2(v1[1],v1[0])

        v2 = np.array([np.cos(theta), np.sin(theta)])

        p1t = pos_start + v2 * r_start

        v4 = np.array([np.cos(theta + n * np.pi), np.sin(theta + n * np.pi)]) * r_end

        p2t = pos_end + v4

        if dir1 == DubinsSegmentType.LEFT:
            sgn = -1
        elif dir1 == DubinsSegmentType.RIGHT:
            sgn = +1
        
        dir = theta - np.pi/2 * sgn
        start_config = SE2Transform(p1t,dir)
        end_config = SE2Transform(p2t,dir)

    elif circle_start.gear == Gear.REVERSE and circle_end.gear == Gear.REVERSE:
        circle_end_rev = circle_end
        circle_end_rev.gear = Gear.FORWARD
        circle_start_rev = circle_start
        circle_start_rev.gear = Gear.FORWARD

        tan = calculate_tangent_btw_circles(circle_end_rev, circle_start_rev)
        if not tan:
            return []
        start_config = tan[0].end_config
        end_config = tan[0].start_config
        circle_start.gear = Gear.REVERSE

    return [Line(start_config, end_config, circle_start.gear)]  # [] i.e., [Line(),...]

def compute_arc_angles(curve_type: DubinsSegmentType, start: SE2Transform, end: SE2Transform, gear):

    arc_angle = (mod_2_pi(end.theta) - mod_2_pi(start.theta))
    
    # compution of arc angle depend on direction of rotation
    if curve_type == DubinsSegmentType.RIGHT and arc_angle>0:
        arc_angle = arc_angle - 2 * np.pi
        
    elif curve_type == DubinsSegmentType.LEFT and arc_angle<0:
        arc_angle = arc_angle + 2 * np.pi
    
    return abs(arc_angle)

def generate_path(start_curve: Curve, end_curve: Curve, line: Line, start: SE2Transform, end: SE2Transform, radius: float):
    distance = line.length
    start_line = line.start_config
    end_line = line.end_config
    
    arc_angle1 = compute_arc_angles(start_curve.type, start, start_line, start_curve.gear)
    arc_angle2 = compute_arc_angles(end_curve.type, end_line, end, start_curve.gear)

    # from start point to start of tangent
    first = Curve(start_config=start, end_config=start_line, center=start_curve.center, radius=radius, curve_type=start_curve.type, arc_angle=arc_angle1, gear=Gear.FORWARD)
    # from end of tangent to end point
    second = Curve(start_config=end_line, end_config=end, center=end_curve.center, radius=radius, curve_type=end_curve.type, arc_angle=arc_angle2, gear=Gear.FORWARD)


    lenght = first.length + distance + second.length
    Path = [first, line, second]

    return lenght, Path

def generate_path_reeds(start_curve_reversed: Curve, end_curve_reversed: Curve, line_reversed: Line, start_reversed: SE2Transform, end_reversed: SE2Transform, radius: float, gear: Gear):
    distance = line_reversed.length
    start_line = line_reversed.start_config
    end_line = line_reversed.end_config
    
    arc_angle1 = compute_arc_angles(start_curve_reversed.type, start_line, start_reversed, start_curve_reversed.gear)
    arc_angle2 = compute_arc_angles(end_curve_reversed.type, end_reversed, end_line, start_curve_reversed.gear)

    # from start_reversed point to end of tangent
    first = Curve(start_config=start_reversed, end_config=start_line, center=start_curve_reversed.center, radius=radius, curve_type=start_curve_reversed.type, arc_angle=arc_angle1, gear=gear)
    # from end of tangent to end point
    second = Curve(start_config=end_line, end_config=end_reversed, center=end_curve_reversed.center, radius=radius, curve_type=end_curve_reversed.type, arc_angle=arc_angle2, gear=gear)

    lenght = first.length + distance + second.length
    #new_line = Line(line_reversed.end_config, line_reversed.start_config, gear)
    Path = [second, line_reversed, first]

    return lenght, Path

def generate_middle_circles(start_curve: Curve, end_curve: Curve, radius):
    
    assert start_curve.type == end_curve.type
    
    circles = []

    p1 = start_curve.center.p
    p2 = end_curve.center.p
    D = distance_between_points(p1, p2)
    v1 = p2 - p1

    theta = np.arccos(D/(4*radius))
    phi = [theta + np.arctan2(v1[1], v1[0]), - theta + np.arctan2(v1[1], v1[0])]

    for angle in phi:
        p3 = np.array([p1[0] + 2 * radius * np.cos(angle), p1[1] + 2 * radius * np.sin(angle)])

        v2 = p1 - p3
        norm = distance_between_points(p1, p3)
        v2 = v2 * radius/norm

        p1t = p3 + v2

        v3 = p3 - p2
        norm = distance_between_points(p2, p3)
        v3 = v3 * radius/norm

        p2t = p2 + v3

        if start_curve.type == DubinsSegmentType.RIGHT:
            curve_type = DubinsSegmentType.LEFT
            n = -1
        else:
            curve_type = DubinsSegmentType.RIGHT
            n = +1
        
        start = SE2Transform(p1t, mod_2_pi(angle + n * np.pi/2))
        end = SE2Transform(p2t, mod_2_pi(np.arctan2(v3[1],v3[0]) + n * np.pi/2))
        centre = SE2Transform(p3, 0)

        arc_angle = compute_arc_angles(curve_type, start, end, start_curve.gear)

        curve = Curve(start_config=start, end_config=end, center=centre, radius=radius, curve_type=curve_type, arc_angle=arc_angle, gear=Gear.FORWARD)

        circles.append(curve)

    return circles

def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!
    # try all of them and pick the best (shortest) one
    lenght = []
    Path = []

    # generate circles
    start_circles = calculate_turning_circles(start_config, radius)
    start_left = start_circles.left
    start_right = start_circles.right

    end_circles = calculate_turning_circles(end_config, radius)
    end_left = end_circles.left
    end_right = end_circles.right

    for start in [start_right, start_left]:
        for end in [end_right, end_left]:
            # CSC trajectories
            # compute tangent
            tan = calculate_tangent_btw_circles(start, end)
            if len(tan) > 0:
                current_lenght, current_path = generate_path(start, end, tan[0], start_config, end_config, radius)
                lenght.append(current_lenght)
                Path.append(current_path)

            # CCC trajectories
            if (distance_between_points(start_config.p, end_config.p) < 4 * radius) and (start.type == end.type):
                # generate middle circles
                for circles in generate_middle_circles(start, end, radius):
                    arc_angle1 = compute_arc_angles(start.type, start.start_config, circles.start_config, start.gear)
                    arc_angle2 = compute_arc_angles(end.type, circles.end_config, end.end_config, end.gear)

                    # from start point to start of tangent
                    first = Curve(start_config=start.start_config, end_config=circles.start_config, center=start.center, radius=radius, curve_type=start.type, arc_angle=arc_angle1)
                    # from end of tangent to end point
                    second = Curve(start_config=circles.end_config, end_config=end.end_config, center=end.center, radius=radius, curve_type=end.type, arc_angle=arc_angle2)

                    lenght.append(first.length + circles.length + second.length)
                    Path.append([first, circles, second])

    # check min lenght
    number = min(lenght)
    Path = Path.pop(lenght.index(number))

    return Path  # e.g., [Curve(), Line(),..]

def compute_length(path):
    length = 0
    for element in path:
        length += element.length
    
    return length

def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
    # try all of them and pick the best (shortest) one
    lenght = []
    Path = []

    # generate circles
    start_circles = calculate_turning_circles(start_config, radius)
    start_left = start_circles.left
    start_right = start_circles.right

    end_circles = calculate_turning_circles(end_config, radius)
    end_left = end_circles.left
    end_right = end_circles.right

    # considering all dubins paths
    dubins_optimal_path = calculate_dubins_path(start_config, end_config, radius)
    length_dubins = compute_length(dubins_optimal_path)
    
    lenght.append(length_dubins)
    Path.append(dubins_optimal_path)

    for start in [start_right, start_left]:
        for end in [end_right, end_left]:  
            # TO COMPLETE -- NOT WORKING
            # C-S-C- trajectories
            # reverse circles
            start.gear = Gear.REVERSE
            end.gear = Gear.REVERSE
            tan = calculate_tangent_btw_circles(end, start)
            if len(tan) > 0:
                #line=Line(tan[0].start_config, tan[0].end_config, Gear.FORWARD)
                #current_lenght, current_path = generate_path_reeds(start, end, tan[0], start_config, end_config, radius, Gear.REVERSE)
                mid_line = tan[0]
                theta_1 = mid_line.start_config.theta
                theta_2 = mid_line.end_config.theta

                arc_angle_1 = -1*start.type*(theta_1 - start_config.theta)
                arc_angle_2 = -1*end.type*(end_config.theta - theta_2)

                s_curve = Curve(start_config, mid_line.start_config,
                                start.center, start.radius, start.type,
                                arc_angle_1, Gear.REVERSE)

                e_curve = Curve(mid_line.end_config, end_config,
                                end.center, end.radius, end.type,
                                arc_angle_2, Gear.REVERSE)

                current_path = [s_curve, mid_line, e_curve]
                    
                lenght.append(compute_length(current_path))
                Path.append(current_path)

    # check min lenght
    number = min(lenght)
    Path = Path.pop(lenght.index(number))

    return Path  # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]