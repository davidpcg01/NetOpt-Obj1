import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple

class Storage:
    def __init__(self, name: str, lon: float = None, lat: float = None, capacity: float = 0, cost: float = 0, initalvol: float = 0) -> None:
        self.name = name
        self.lon = lon
        self.lat = lat
        self.capacity = capacity
        self.cost = cost
        self.initialvol = initalvol

    def add_capacity(self, capacity) -> None:
        self.capacity = capacity

    def add_cost(self, cost) -> None:
        self.cost = cost

    def set_name(self, name) -> None:
        self.name = name

    def get_capacity(self) -> float:
        return self.capacity
    
    def get_cost(self) -> float:
        return self.cost
    
    def get_initialvol(self) -> float:
        return self.initialvol
    
    def get_name(self) -> str:
        return self.name
    
    def get_loc(self) -> Tuple[float, float]:
        return self.lat, self.lon
    

if __name__ == '__main__':
    #test storage location
    sto1 = Storage('sto1', -94.1, 106, 1000, 2)

    #print results
    print(sto1.get_name(), sto1.get_loc(), sto1.get_capacity(), sto1.get_cost())
