import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple

class TransportConn():
    def __init__(self, name: str = None,  conntype: str = 'pipeline', coststructure: str = 'fixed', upstream: str = None, 
                 downstream: str = None, bidirectional: bool = False, fixedcost: float = 0, capacity: float = 0.,
                 varpipecost: Dict = None, vartuckcost: Dict = None) -> None:
        self.name = name
        self.conntype = conntype
        self.coststructure = coststructure
        self.upstream = upstream
        self.downstream = downstream
        self.isbidirectional = bidirectional
        self.fixedcost = fixedcost
        self.varpipecost = varpipecost
        self.vartruckcost = vartuckcost
        self.capacity = capacity


        self._sanity_check()

        self._is_fixed()
        self._is_pipeline()

    
    def _add_cost(self, cost, costtype: str = 'fixed') -> None:
        if costtype == 'fixed':
            assert(type(cost) in [float, int])
            self.fixedcost = cost
        else:
            assert(type(cost) is dict)
            if self.ispipeline:
                self.varpipecost = cost
            else:
                self.vartruckcost = cost
        
        self.coststructure = costtype
        self._is_fixed()

    def set_upstream(self, upstream: str) -> None:
        self.upstream = upstream

    def set_downstream(self, downstream: str) -> None:
        self.downstream = downstream

    def set_conntype(self, conntype: str) -> None:
        self.conntype = conntype

    def set_name(self, name: str) -> None:
        self.name = name

    def set_ends(self, upstream: str, downstream:str) -> None:
        self.upstream = upstream
        self.downstream = downstream

    def set_bidirectional(self, bidirectional = True):
        self.isbidirectional = bidirectional

    def get_ends(self) -> Tuple[str,str]:
        return (self.upstream, self.downstream)
                
    def get_upstream(self) -> str:
        return self.upstream
    
    def get_downstream(self) -> str:
        return self.downstream
    
    def get_conntype(self) -> str:
        return self.conntype
    
    def get_name(self) -> str:
        return self.name
    
    def get_coststructure(self):
        return self.coststructure

    def _is_bidirectional(self) -> bool:
        return self.isbidirectional
    
    def _is_fixed(self) -> bool:
        self.isfixed = self.coststructure == 'fixed'
        return self.isfixed
    
    def _is_pipeline(self) -> bool:
        self.ispipeline = self.conntype == 'pipeline'
        return self.ispipeline
    
    def _get_cost(self):
        if self.isfixed:
            return self.fixedcost
        else:
            if self.ispipeline:
                return self.varpipecost
            else:
                return self.vartruckcost
            
    def get_fixed_cost(self):
        return self.fixedcost
    
    def get_capacity(self):
        return self.capacity



    def _sanity_check(self) -> None:
        #sanity check
        if self.conntype not in ['pipeline', 'truck']:
            raise KeyError('Error: Connection Type must either be "pipeline" or "truck".')
        
        if self.coststructure not in ['fixed', 'variable']:
            raise KeyError('Error: Cost structure must either be "fixed" or "variable".')
        
        if self.conntype == 'pipeline':
            if self.vartruckcost is not None:
                raise KeyError('Error: Cannot Input Variable Truck Cost for pipeline')
            if (self.coststructure == 'fixed') and (self.varpipecost is not None):
                raise KeyError('Error: Cannot Input Variable Pipeline Costs for fixed cost structure')
            
        if self.conntype == 'truck':
            if self.varpipecost is not None:
                raise KeyError('Error: Cannot Input Variable Pipeline Cost for Truck')
            if (self.coststructure == 'fixed') and (self.vartruckcost is not None):
                raise KeyError('Error: Cannot Input Variable Truck Costs for fixed cost structure')

            
        
if __name__ == '__main__':
    #create sample pipeline
    conn1 = TransportConn("conn1", conntype='pipeline')
    conn1._add_cost(5.4, costtype='fixed')
    conn1.set_upstream('S1')
    conn1.set_downstream('D1')

    #conn2
    conn2 = TransportConn()
    conn2.set_name('conn2')
    conn2.set_conntype('truck')
    cost = {'0-10':20, '11-20':23, '21-40':26, '40+':30}
    conn2._add_cost(cost=cost, costtype='variable')
    conn2.set_ends(upstream='S2', downstream='D2')
    conn2.set_bidirectional()

    #print values
    print(conn1.get_name(), conn1.get_conntype(), conn1.get_coststructure(), conn1.get_ends(), conn1._get_cost())
    print(conn2.get_name(), conn2.get_conntype(), conn2.get_coststructure(), conn2.get_upstream(), conn2.get_downstream(), conn2._get_cost())

    print(conn2.get_ends())