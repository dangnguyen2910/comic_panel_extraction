import numpy as np 
import math

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.a = y2 - y1
        self.b = x1 - x2
        self.c = (y1 - y2) * x1 + (x1 - x2) * y1

        self.orientation = self.set_orientation()
        self.length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 )

    def set_orientation(self):
        angle = self.calculate_angle()
        if angle > 180: 
            angle = angle - 180
        
        if angle > 90: 
            angle = 180 - angle

        if angle >= 45:
            return "vertical"
        else:
            return "horizontal"
    
        
    def calculate_angle(self):
        ox = (1,0)
        A = (self.x1, self.y1)
        B = (self.x2, self.y2)
        AB = (self.x2 - self.x1, self.y2 - self.y1)
        mag_AB = np.sqrt(AB[0]**2 + AB[1]**2)
        dot_prod = np.dot(AB, ox)
        mag_prod = mag_AB * 1
        angle = math.acos(dot_prod / mag_prod)  
        return math.degrees(angle) % 360