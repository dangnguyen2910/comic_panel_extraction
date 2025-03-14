from post_process import * 
import numpy as np 

mask = np.array([
        [1,0,1], 
        [1,1,1], 
        [0,0,0]
    ])

def test_calculate_ratio_horizontal():
    
    point1 = (0,0)
    point2 = (0,2)
    assert calculate_ratio_of_color_pixels_between_two_points(mask, point1, point2) == 2/3

def test_calculate_ratio_vertical():

    point1 = (0,0)
    point2 = (2,0)
    assert calculate_ratio_of_color_pixels_between_two_points(mask, point1, point2) == 2/3

def test_calculate_ratio_diagonal():

    point1 = (0,0)
    point2 = (2,2)
    assert calculate_ratio_of_color_pixels_between_two_points(mask, point1, point2) == 2/3

def test_calculate_ratio_diagonal2():

    point1 = (2,0)
    point2 = (0,2)
    assert calculate_ratio_of_color_pixels_between_two_points(mask, point1, point2) == 2/3

def test_calculate_ratio_random():

    point1 = (0,0)
    point2 = (1,2)
    assert calculate_ratio_of_color_pixels_between_two_points(mask, point1, point2) == 2/3

def test_is_legit1():
    point1 = (0,0)
    point2 = (0,2)
    assert is_legit(mask, point1, point2, 0.5) == True

def test_is_legit2():
    point1 = (0,0)
    point2 = (0,2)
    assert is_legit(mask, point1, point2, 0.7) == False


def test_distance():
    x0, y0 = (0,0)
    x1, y1 = (5,2)
    assert distance((x0, y0), (x1, y1)) == np.sqrt(29)


def test_classLine():
    line = Line(0,0,1,1)
    assert line.orientation == 'vertical'
    assert line.length == np.sqrt(2)
    assert line.a == 1 and line.b == -1 or line.a == -1 and line.b == 1
    assert line.c == 0


def test_calculate_intersection():
    line1 = Line(0,0,2,2)
    # line2 = Line()


def test_distance_from_point_to_line():
    line = Line(0,0, 2,0)
    assert distance_from_point_to_line((0,2), line) == 2


def test_calculate_intersection1(): 
    line1 = Line(0,0, 2,0)
    line2 = Line(0,0, 0,2)
    assert calculate_intersection(line1, line2) == (0,0)


def test_calculate_intersection2():
    line1 = Line(0,0,2,0)
    line2 = Line(0,1,0,3) 
    assert line1.length == 2
    assert calculate_intersection(line1, line2) == (0,0)

def test_calculate_intersection3():
    line1 = Line(0,1,0,3)
    line2 = Line(1,0,2,0)
    assert calculate_intersection(line1, line2) == None

def test_distance_2_points():
    assert distance_2_points((0,0), (0,1)) == 1
    assert distance_2_points((0,0), (0,3)) == 3
    assert distance_2_points((0,0), (1,0)) == 1
    assert distance_2_points((0,0), (2,0)) == 2