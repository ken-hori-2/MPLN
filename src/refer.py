
from lib2to3.pytree import Node
from sre_parse import State
import numpy as np
from itertools import count
class Property():
    def __init__(self, *arg):

        
        self.pre = np.array([
                            [3, "g"],
                            [3, "K"],
                            [3, "J"],
                            [3, "I"],
                            [3, "H"],
                            [3, "O"],
                            [3, "F"],
                            [3, "E"],
                            [3, "D"],
                            [3, "C"],
                            [3, "B"],
                            [3, "A"],
                            [0, "s"]
                            ])

    
    def reference(self):
        
        
        Node = self.pre[:, 1]
        Arc = self.pre[:, 0]
        Node = Node.tolist()
        Arc = Arc.tolist()
        num = [float(i) for i in Arc]
        Arc_sum = sum(num)

        PERMISSION = [
                [0],
                [3],
                [6],
                [9],
                [12],
                [15],
                [18],
                [21],
                [24]
        ]

        return self.pre, Node, Arc, Arc_sum, PERMISSION

if __name__ == "__main__":
   test = Property()
   test.reference