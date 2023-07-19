from math import sqrt

def proper_round(num, dec=0) -> float:
    return round(num, dec)

class Point:
    def __init__(self, x=-1, y=-1, cap=0) -> None:
        self.x = x
        self.y = y
        self.cap = cap

    def get_distance(self, other) -> int:
        d = sqrt((other.x -  self.x) * (other.x -  self.x) +
                    (other.y -  self.y) * (other.y -  self.y))
        return int(proper_round(d))
    
    def __lt__(self, other) -> bool:
        return self.cap < other.cap
    
    def __sub__(self, other):
        return Point(self.x, self.y, self.cap - other)

    def __isub__(self, other):
        return self.__sub__(other=other)
    
    def __add__(self, other):
        return Point(self.x, self.y, self.cap + other)
    
    def __iadd__(self, other):
        return self.__add__(other=other)

    def __str__(self) -> str:
        return 'x: ' + str(self.x) + '\ny: ' + str(self.y) + '\ncap: ' + str(self.cap)
