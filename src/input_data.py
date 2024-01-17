import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple
from demand_sink import DemandSink
from supply_source import SupplySource
from transport_conn import TransportConn
from storage import Storage
import networkx as nx
import matplotlib.pyplot as plt


class InputData:
    def __init__(self, sourceObjDict: Dict[str, SupplySource] = None, sinkObjDict: Dict[str, DemandSink] = None, 
                 connObjDict: Dict[str, TransportConn] = None, stoObjDict:Dict[str, Storage] = None) -> None:
        self.sourceObjDict = sourceObjDict
        self.sinkObjDict = sinkObjDict
        self.connObjDict = connObjDict
        self.stoObjDict = stoObjDict

        self._extract_pipe_nodes()

        self._pipe_name_to_arc()
        self._arc_to_pipe_name()

        self.generate_network()
    
    def _validate_source_data(self):
        if self.sourceObjDict is not None:
            print('VALIDATING INPUT SOURCES...')
            for key, source in self.sourceObjDict.items():
                print(f'Evaluating Source {key}')
                if source._hasforecast():
                    print('     Valid Supply Forecast')
                else:
                    raise KeyError(f'Supply forecast for {key} is invalid, please check input data')
            print('SOURCE DATA VALIDATION COMPLETE')
            print("")
        else:
            raise KeyError('Source Data Dictionary is Not Provided')

    def _validate_sink_data(self):
        if self.sinkObjDict is not None:
            print('VALIDATING INPUT SINKS...')
            for key, sink in self.sinkObjDict.items():
                print(f'Evaluating Sink {key}')
                if sink._hasforecast():
                    print('     Valid Demand Forecast')
                else:
                    raise KeyError(f'Demand forecast for {key} is invalid, please check input data')
            print('SINK DATA VALIDATION COMPLETE')
            print("")
        else:
            raise KeyError('Sink Data Dictionary is Not Provided')
        pass

    def _validate_conn_data(self):
        pass

    def _valide_sto_data(self):
        pass

    def validate_input(self):
        self._validate_source_data()
        self._validate_sink_data()
        self._validate_conn_data()
        self._valide_sto_data()

    def _pipe_name_to_arc(self):
        self.pipe_to_arc = {key:self.connObjDict[key].get_ends() for key in self.connObjDict.keys()}

    def _arc_to_pipe_name(self):
        self.arc_to_pipe = {val:key for key, val in self.pipe_to_arc.items()}

    def _extract_pipe_nodes(self):
        src_sink_sto = list(self.sourceObjDict.keys()) + list(self.sinkObjDict.keys()) + list(self.stoObjDict.keys())
        src_sink_sto = set(src_sink_sto)
        all_arcs = []
        for arc in self.connObjDict.values():
            node1,node2 = arc.get_ends()
            all_arcs.append(node1)
            all_arcs.append(node2)
        all_arcs = set(all_arcs)
        
        self.pipe_nodes = all_arcs - src_sink_sto

    def generate_network(self) -> None:
        self.G = nx.Graph()
        self.G.add_nodes_from(self.sourceObjDict.keys())
        self.G.add_nodes_from(self.sinkObjDict.keys())
        self.G.add_nodes_from(self.stoObjDict.keys())

        for edge in self.connObjDict.values():
            # print((edge.get_ends()))
            self.G.add_edges_from([(edge.get_ends())])
    
    def plot_network(self):
        # subax1 = plt.subplot(121)
        nx.draw(self.G, with_labels=True, font_weight='bold')
        plt.show()

    def get_assets(self):
        self.generate_network()
        return list(self.G.nodes())
    
    def get_sources(self):
        return list(self.sourceObjDict.keys())
    
    def get_sinks(self):
        return list(self.sinkObjDict.keys())
    
    def get_storage(self):
        return list(self.stoObjDict.keys())

    def get_nodes(self):
        return self.pipe_nodes
    
    def get_pipe_to_arc(self):
        return self.pipe_to_arc
    
    def get_arc_to_pipe(self):
        return self.arc_to_pipe
    
    def get_arcs(self):
        res = []
        for conn in self.connObjDict.values():
            res.append(conn.get_ends())
        return res
    
    def get_time(self):
        all_times = set()
        for src in self.sourceObjDict.values():
            start, end = src.get_dates()
            all_times.add(start)
            all_times.add(end)
        for sink in self.sinkObjDict.values():
            start, end = sink.get_dates()
            all_times.add(start)
            all_times.add(end)
        return sorted(all_times)
        





if __name__ == '__main__':
    #create some test Source Data and save it into a source Object dict
    #source1: constant supply 500bwpd
    so1 = SupplySource("prd1", -98.1, 104.2)
    so1._add_constant_supply(500, '2020-01-01', '2022-12-31')
    so1._add_cost(5, ctype='revenue')
    #source2: constant supply 1000bwpd
    so2 = SupplySource('prd2', -99.05, 104.36)
    so2._add_constant_supply(1000, '2019-06-01', '2023-08-01')
    so2._add_cost(0.5, ctype='cost')

    #create the source object dict
    sourceObjDict = {}
    sourceObjDict[so1._get_name()] = so1
    sourceObjDict[so2._get_name()] = so2
    

    #create some test demand data and save it into a sink object dict
    #sink1: 
    si1 = DemandSink("disp1", -96.2, 106)
    si1._add_constant_forecast(450, '2020-03-01', '2023-01-01')
    si1._add_cost(7)

    #sink2:
    si2 = DemandSink("disp2", -94.5, 103)
    si2._add_constant_forecast(800, '2019-07-06', '2023-07-01')
    si2._add_cost(4)

    sinkObjDict = {}
    sinkObjDict[si1._get_name()] = si1
    sinkObjDict[si2._get_name()] = si2

    #create test storage
    #sto1:
    sto1 = Storage('sto1', -94.5, 101, 10000, cost=3)
    #sto2:
    sto2 = Storage('sto2', -94.9, 103, 12000, cost=0.5)
    
    stoObjDict = {}
    stoObjDict[sto1.get_name()] = sto1
    stoObjDict[sto2.get_name()] = sto2
    

    #create pipelines
    #pipe1:
    pi1 = TransportConn('pipe1', upstream='prd1', downstream='sto1', fixedcost=2)
    #pipe2:
    pi2 = TransportConn('pipe2', upstream='sto1', downstream='disp1', fixedcost=3)
    #pipe3:
    pi3 = TransportConn('pipe3', upstream='prd2', downstream='disp2', fixedcost=5)
    #pipe4:
    pi4 = TransportConn('pipe4', upstream='prd2', downstream='sto2', fixedcost=2.5)
    #pipe5:
    pi5 = TransportConn('pipe5', upstream='sto2', downstream='disp2', fixedcost=3.5)
    #pipe6:
    pi6 = TransportConn('pipe6', upstream='sto1', downstream='sto2', fixedcost=0.3)
    #pipe7
    pi7 = TransportConn('pipe7', upstream='prd1', downstream='node1', fixedcost=0.1)
    #pipe8
    pi8 = TransportConn('pipe8', upstream='node1', downstream='sto2', fixedcost=0.1)

    connObjDict = {}
    connObjDict[pi1.get_name()] = pi1
    connObjDict[pi2.get_name()] = pi2
    connObjDict[pi3.get_name()] = pi3
    connObjDict[pi4.get_name()] = pi4
    connObjDict[pi5.get_name()] = pi5
    connObjDict[pi6.get_name()] = pi6
    connObjDict[pi7.get_name()] = pi7
    connObjDict[pi8.get_name()] = pi8


    #create
    inp = InputData(sourceObjDict=sourceObjDict, sinkObjDict=sinkObjDict, stoObjDict=stoObjDict, connObjDict=connObjDict)
    inp.validate_input()
    # inp.generate_network()

    print(inp.get_arc_to_pipe())
    inp._extract_pipe_nodes()
    inp.plot_network()