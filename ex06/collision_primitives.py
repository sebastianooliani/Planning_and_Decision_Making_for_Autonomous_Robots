from pdm4ar.exercises_def.ex06.structures import *
import numpy as np
import triangle as tr

class CollisionPrimitives:
    """
    Class of collusion primitives
    """
    @staticmethod
    def compute_distance(start, end) -> float:
        return ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5

    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        # get the center coordinate
        center=np.array([c.center.x,c.center.y])
        point=np.array([p.x,p.y])
        # compute distance from center
        # if distance<=radius => collision
        if CollisionPrimitives.compute_distance(center, point) <= c.radius:
            return True
        else:
            return False

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        x1=t.v1.x
        x2=t.v2.x
        x3=t.v3.x
        y1=t.v1.y
        y2=t.v2.y
        y3=t.v3.y
        area_orig = np.round(abs( (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) ),10)
        px=p.x
        py=p.y
        area1 = abs( (x1-px)*(y2-py) - (x2-px)*(y1-py) )
        area2 = abs( (x2-px)*(y3-py) - (x3-px)*(y2-py) )
        area3 = abs( (x3-px)*(y1-py) - (x1-px)*(y3-py) )
        # check if point is inside the triangle area
        if (np.round(area1 + area2 + area3, 10) == area_orig):
            return True
        else:
            return False

    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        # divide polygon in triangle
        V=[]
        for i in range(0,len(poly.vertices)):
            V.append([poly.vertices[i].x,poly.vertices[i].y])
        # create dictionary
        t = tr.triangulate({'vertices': V}, 'a0.2')
        # triangle with its vertices coordinates
        triangle = np.array(t['triangles'])
        # coordinates of each vertix of the triangle
        vert = np.array(t['vertices'])

        # for each vertix create the triangle
        tri=[]
        for element in triangle:
            tri.append(Triangle(Point(vert[element[0]][0], vert[element[0]][1]),\
                 Point(vert[element[1]][0], vert[element[1]][1]), Point(vert[element[2]][0], vert[element[2]][1])))
        # check for each triangle if point is inside
        for element in tri:
            # if there is a collision, change boolean
            if CollisionPrimitives.triangle_point_collision(element, p) is True:
                return True
        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        # check if extremes are in circle
        p1 = segment.p1
        p2 = segment.p2
        if CollisionPrimitives.circle_point_collision(c, p1) or CollisionPrimitives.circle_point_collision(c, p2):
            return True

        # distance between two points
        d = CollisionPrimitives.compute_distance([p2.x, p2.y], [p1.x, p1.y])
        # get the closest point to the line
        alpha =((p2.x - p1.x) * (c.center.x - p1.x) + \
            (p2.y - p1.y) * (c.center.y - p1.y))/d**2
        # M = closest point to the line
        M = np.array([p1.x + (p2.x - p1.x) * alpha, \
            p1.y + (p2.y - p1.y) * alpha])
        # centre of the circle
        centre = np.array([c.center.x, c.center.y])
        if CollisionPrimitives.compute_distance(centre, M) > c.radius:
            # no intersection
            return False
        elif CollisionPrimitives.compute_distance(centre, M) <= c.radius:
            # check if M is in the segment
            if CollisionPrimitives.compute_distance(M, [p1.x, p1.y]) <= d\
                 and CollisionPrimitives.compute_distance(M, [p2.x, p2.y]) <= d:
                # circle and line intersect
                return True
            # if M is not in the segment
            elif CollisionPrimitives.compute_distance(centre, [p1.x, p1.y]) <= c.radius\
                 or CollisionPrimitives.compute_distance(centre, [p2.x, p2.y]) <= c.radius:
                # circle and segment intersect
                return True
            else:
                # no intersection
                return False
        
    @staticmethod
    def ccw(A,B,C):
        return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

    @staticmethod
    # Return true if line segments AB and CD intersect
    def intersect(A,B,C,D):
        return CollisionPrimitives.ccw(A,C,D) != CollisionPrimitives.ccw(B,C,D)\
             and CollisionPrimitives.ccw(A,B,C) != CollisionPrimitives.ccw(A,B,D)
    
    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        # check extremes
        p1 = segment.p1
        p2 = segment.p2
        if CollisionPrimitives.triangle_point_collision(t, p1) \
            or CollisionPrimitives.triangle_point_collision(t, p2):
            return True
        # verify if there are interesections
        if CollisionPrimitives.intersect(t.v1, t.v2, p1, p2) \
            or CollisionPrimitives.intersect(t.v1, t.v3, p1, p2) \
                or CollisionPrimitives.intersect(t.v3, t.v2, p1, p2):
                return True
        else:
            return False

    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        # divide polygon in triangle
        V=[]
        for i in range(0,len(p.vertices)):
            V.append([p.vertices[i].x, p.vertices[i].y])
        # create dictionary
        t=tr.triangulate({'vertices': V}, 'a0.2')
        # triangle with its vertices coordinates
        triangle = np.array(t['triangles'])
        # coordinates of each vertix of the triangle
        vert = np.array(t['vertices'])

        # for each vertix create the triangle
        tri=[]
        for element in triangle:
            tri.append(Triangle(Point(vert[element[0]][0], vert[element[0]][1]), Point(vert[element[1]][0], vert[element[1]][1]), Point(vert[element[2]][0], vert[element[2]][1])))
        
        # check collision with segment
        for element in tri:
            # if there is a collision, change boolean
            if CollisionPrimitives.triangle_segment_collision(element, segment) is True:
                return True
        return False

    @staticmethod
    def polygon_segment_collision_aabb(p: Polygon, segment: Segment) -> bool:
        aabb = CollisionPrimitives._poly_to_aabb(p)
        # get vertices of a rectangle out of p_min and p_max
        diagonal = CollisionPrimitives.compute_distance([aabb.p_max.x, aabb.p_max.y], [aabb.p_min.x, aabb.p_min.y])
        v = np.array([aabb.p_max.x - aabb.p_min.x, aabb.p_max.y - aabb.p_min.y])
        theta = np.arctan2(v[1], v[0])
        vertices = [[aabb.p_max.x, aabb.p_max.y], [aabb.p_min.x, aabb.p_min.y],\
            [aabb.p_min.x + diagonal * np.cos(theta), aabb.p_min.y],\
                [aabb.p_min.x, aabb.p_min.y + diagonal * np.sin(theta)]]
        
        # divide polygon in triangle
        V=[]
        for i in range(0,len(vertices)):
            V.append([vertices[i][0], vertices[i][1]])
        # create dictionary
        t=tr.triangulate({'vertices': V}, 'a0.2')
        # triangle with its vertices coordinates
        triangle = np.array(t['triangles'])
        # coordinates of each vertix of the triangle
        vert = np.array(t['vertices'])

        # for each vertix create the triangle
        tri=[]
        for element in triangle:
            tri.append(Triangle(Point(vert[element[0]][0], vert[element[0]][1]), Point(vert[element[1]][0], vert[element[1]][1]), Point(vert[element[2]][0], vert[element[2]][1])))
        
        # check collision with segment
        for element in tri:
            # if there is a collision, change boolean
            if CollisionPrimitives.triangle_segment_collision(element, segment) is True:
                return True
        
        # else, standard code
        return CollisionPrimitives.polygon_segment_collision(p, segment)

    @staticmethod
    def _poly_to_aabb(g: Polygon) -> AABB:
        # todo feel free to implement functions that upper-bound a shape with an
        #  AABB or simpler shapes for faster collision checks
        return AABB(p_min=Point(0, 0), p_max=Point(1, 1))