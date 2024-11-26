import numpy as np 

def distance_2_points(point1, point2) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2)**2 + (y1-y2)**2)



def calculate_intersection(line1, line2) -> tuple:
    x1, y1, x2, y2 = line1.x1, line1.y1, line1.x2, line1.y2
    x3, y3, x4, y4 = line2.x1, line2.y1, line2.x2, line2.y2

    denominator = (line1.x1 - line1.x2) * (line2.y1 - line2.y2) - (line1.y1 - line1.y2) * (line2.x1 - line2.x2)
    if denominator == 0:  
        return None # Parallel lines
    
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denominator
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denominator

    px = int(px)
    py = int(py)
        
    return (px, py)



def find_intersection(vertical_lines, horizontal_lines) -> list[tuple]:
    point_list = []
    for vline in vertical_lines: 
        for hline in horizontal_lines:
            points = calculate_intersection(vline, hline)
            point_list.append(points)

    return point_list
