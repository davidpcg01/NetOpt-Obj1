import pandas as pd
import numpy as np
import gurobipy as gp
import networkx as nx
import pulp as pl
from pulp import *
from gurobipy import GRB
from typing import List, Tuple, Dict, Set
from demand_sink import DemandSink
from supply_source import SupplySource
from transport_conn import TransportConn
from storage import Storage
from input_data import InputData
import os




class Mathmodel():
    def __init__(self, name: str, optType: str, sourceObjDict: Dict[str, SupplySource] = None, sinkObjDict: Dict[str, DemandSink] = None, 
                 connObjDict: Dict[str, TransportConn] = None, stoObjDict:Dict[str, Storage] = None) -> None:
        self.name = name
        self.optType = optType

        self.InputData = InputData(sourceObjDict=sourceObjDict, sinkObjDict=sinkObjDict,
                                   connObjDict=connObjDict, stoObjDict=stoObjDict)
        
        self.InputData.validate_input()
        self._initialize_sets()
        self._initialize_source_parameters()
        self._initialize_sink_parameters()
        self._initialize_storage_parameters()
        self._initialize_transport_parameters()
        self._initialize_slack_parameters()

        self.vars: Dict[str, gp.tupledict] = {}
        self.cons: Dict[str, gp.tupledict] = {}

        self.BigM = 1e6

        self.MPS_FILE_PATH = os.path.join(f"WaterOpt/model_data/{self.name}.mps")
        self.LP_FILE_PATH = os.path.join(f"WaterOpt/model_data/{self.name}.lp")
        self.SOL_FILE_PATH = os.path.join(f"WaterOpt/model_data/{self.name}.sol")
        self.ILP_FILE_PATH = os.path.join(f"WaterOpt/model_data/{self.name}.ilp")
        

    def _initialize_sets(self) -> None:
        self.assets: Set = set() #all assets which includes all sources, sinks, storage tanks and nodes
        self.src: Set = set() #all source nodes
        self.sink: Set = set() #all sink nodes
        self.storage: Set = set() #all storage tanks nodes
        self.nodes: Set = set() #all pipeline nodes
        self.time: Set = set() #all time periods
        self.a_a: Set = set() #all node to node connections #TODO: account for bidirectional connections, strategy is all conn with node up, down have bidir, all pipe to sto or sto to pipe or sto to sto have bidirec or dem to dem. 
        #note: flow out of source is unidirectional, so we dont have backflow

    def _initialize_source_parameters(self) -> None:
        self.src_cost: Dict = {} #cost of produced water (could be revenue)
        self.src_forecast: Dict = {} #capacity of source

    def _initialize_sink_parameters(self) -> None:
        self.sink_cost: Dict = {} #cost of water inj or disposal
        self.sink_cap: Dict = {} #disposal well capacity

    def _initialize_storage_parameters(self) -> None:
        self.storage_cost: Dict = {} #cost of storage
        self.storage_cap: Dict = {} #capacity
        self.initial_vol: Dict = {} #initial volume

    def _initialize_transport_parameters(self) -> None:
        self.trans_cost: Dict = {} #cost of transport
        self.trans_cap: Dict = {} #pipeline capacity

    def _initialize_slack_parameters(self) -> None: #TODO: add slack costs
        self.slack_demand_cost: float = 1000
        self.slack_supply_cost: float = 1000

    def _generate_sets(self) -> None:
        self.assets = set(self.InputData.get_assets())
        self.src = set(self.InputData.get_sources())
        self.sink = set(self.InputData.get_sinks())
        self.storage = set(self.InputData.get_storage())
        self.nodes = set(self.InputData.get_nodes()) #TODO:create function in input data
        self.a_a = set(self.InputData.get_arcs())

        all_times = self.InputData.get_time()
        #convert data to datetime
        start_date = pd.to_datetime(all_times[0])
        end_date = pd.to_datetime(all_times[-1])

        #get date range
        date_range = pd.date_range(start_date, end_date)
        self.time = [date.strftime('%Y-%m-%d') for date in date_range]

    def _generate_parameters(self):
        #source data
        self.src_cost = {src:{time:self.InputData.sourceObjDict[src]._get_cost() for time in self.time} for src in self.src} #a function of time
        self.src_forecast = {src:{time:0 for time in self.time} for src in self.src} #can change with time, for any time period not in consideration, defaults to zero. This is the suppluy forecast
        for src in self.src:
            forecast_dict = self.InputData.sourceObjDict[src]._get_all_forecast()
            for time in forecast_dict.keys():
                time_str = time.strftime('%Y-%m-%d')
                self.src_forecast[src][time_str] = forecast_dict[time]

        #sink data
        self.sink_cost = {sink:{time:self.InputData.sinkObjDict[sink]._get_cost() for time in self.time} for sink in self.sink} #is a function of time
        self.sink_cap = {sink:{time:0 for time in self.time} for sink in self.sink} #could be a function of time since the capacity is more like a demand forecast
        for sink in self.sink:
            forecast_dict = self.InputData.sinkObjDict[sink]._get_all_forecast()
            for time in forecast_dict.keys():
                time_str = time.strftime('%Y-%m-%d')
                self.sink_cap[sink][time_str] = forecast_dict[time]

        #storage data
        self.storage_cost = {sto:{time:self.InputData.stoObjDict[sto].get_cost() for time in self.time} for sto in self.storage}  #could be a function of time
        self.storage_cap = {sto:self.InputData.stoObjDict[sto].get_capacity() for sto in self.storage} #not a function of time #TODO: think about how to turn off storage tank
        self.initial_vol = {sto:self.InputData.stoObjDict[sto].get_initialvol() for sto in self.storage} #not a function of time

        #transport data
        arc_to_pipe = self.InputData.get_arc_to_pipe()
        self.trans_cost = {arc:{time:self.InputData.connObjDict[arc_to_pipe[arc]].get_fixed_cost() for time in self.time} for arc in self.a_a} #of the form C[(n1,n2)][t]
        self.trans_cap = {arc:{time:self.InputData.connObjDict[arc_to_pipe[arc]].get_capacity() for time in self.time} for arc in self.a_a} #same form as cost, capacity varies over time to allow turning off pipe connections


    def _validation_checks(self) -> None:
        pass


    def create_sets_and_parameters(self) -> None:
        self._generate_sets()
        self._generate_parameters()
        self._validation_checks()

    def create_variables(self) -> None:
        #CONTINUOUS VARIABLES
        #flow from node i to node j
        index = ((node1, node2, t) for (node1, node2) in self.a_a for t in self.time)
        self.vars['arc_flow'] = self.model.addVars(index, name='arc_flow', lb=0, vtype=GRB.CONTINUOUS)

        #water level in each storage tank
        index = ((sto, t) for sto in self.storage for t in self.time)
        self.vars['tank_level'] = self.model.addVars(index, name='tank_level', lb= 0, vtype=GRB.CONTINUOUS)

        #flow out of source
        index = ((src, t) for src in self.src for t in self.time)
        self.vars['source_output'] = self.model.addVars(index, name='source_output', lb=0, vtype=GRB.CONTINUOUS)

        #flow into sink
        index = ((sink, t) for sink in self.sink for t in self.time)
        self.vars['sink_input'] = self.model.addVars(index, name='sink_input', lb=0, vtype=GRB.CONTINUOUS)

        #SLACK VARIABLES
        #supply slack
        index = ((src, t) for src in self.src for t in self.time)
        self.vars['supply_slack'] = self.model.addVars(index, name='supply_slack', lb=0, vtype=GRB.CONTINUOUS)

        #demand slack
        index = ((sink, t) for sink in self.sink for t in self.time)
        self.vars['demand_slack'] = self.model.addVars(index, name='demand_slack', lb=0, vtype=GRB.CONTINUOUS)


        #BINARY VARIABLES
        #pipeline arc activation
        index = ((node1, node2, t) for (node1, node2) in self.a_a for t in self.time)
        self.vars['arc_active'] = self.model.addVars(index, name='arc_active', vtype=GRB.BINARY)
        

    def _initialize_gurobi(self) -> None:
        self.env = gp.Env(empty=True)
        self.env.start()
        self.model = gp.Model(self.name, env=self.env)

    
    def create_constraints(self) -> None:
        self._supply_cons()
        self._demand_cons()
        self._storage_cons()
        self._conn_cons()
        self._node_cons()
        pass

    def _supply_cons(self) -> None:
        supply_to_asset = {s:[a for a in self.assets
                              if (s,a) in self.a_a]
                        for s in self.src}
        asset_to_supply = {s:[a for a in self.assets
                              if (a,s) in self.a_a]
                        for s in self.src}
        
        cons_name = 'supply_capacity'
        constr = (sum(self.vars['arc_flow'][s,a,t] for a in supply_to_asset[s]) 
                  <= self.src_forecast[s][t] + self.vars['supply_slack'][s,t]
                  for s in self.src for t in self.time)
        self.cons[cons_name] = self.model.addConstrs(constr, name=cons_name)
        
        cons_name = 'total_supply'
        constr = (self.vars['source_output'][s,t]
                  == sum(self.vars['arc_flow'][s,a,t] for a in supply_to_asset[s])
                  for s in self.src for t in self.time)
        self.cons[cons_name] = self.model.addConstrs(constr, name=cons_name)


    def _demand_cons(self) -> None:
        asset_to_demand = {d:[a for a in self.assets
                              if (a,d) in self.a_a]
                        for d in self.sink}
        
        cons_name1 = 'demand_capacity'
        constr1 = (sum(self.vars['arc_flow'][a,d,t] for a in asset_to_demand[d]) #+ self.vars['demand_slack'][d,t]
                   <= self.sink_cap[d][t]
                   for d in self.sink for t in self.time)
        # cons_name2 = 'demand_cap_slack'
        # constr2 = (sum(self.vars['arc_flow'][a,d,t] for a in asset_to_demand[d])
        #            <= self.sink_cap[d][t] + self.vars['demand_slack'][d,t]
        #            for d in self.sink for t in self.time)
        cons_name3 = 'total_demand'
        constr3 = (self.vars['sink_input'][d,t]
                 == sum(self.vars['arc_flow'][a,d,t] for a in asset_to_demand[d])
                 for d in self.sink for t in self.time)
        
        self.cons[cons_name1] = self.model.addConstrs(constr1, name=cons_name1)
        # self.cons[cons_name2] = self.model.addConstrs(constr2, name=cons_name2)
        self.cons[cons_name3] = self.model.addConstrs(constr3, name=cons_name3)

    def _storage_cons(self) -> None:
        asset_to_storage = {sto:[a for a in self.assets
                                 if (a, sto) in self.a_a]
                            for sto in self.storage}
        
        storage_to_asset = {sto:[a for a in self.assets
                                 if (sto, a) in self.a_a]
                            for sto in self.storage}
        
        next_t = {self.time[i]:self.time[i-1] for i in range(1, len(self.time))}
        cons_name = 'storage_cap'
        constr = (self.vars['tank_level'][sto,t] 
                  <= self.storage_cap[sto]
                  for sto in self.storage for t in self.time)
        self.cons[cons_name] = self.model.addConstrs(constr, name=cons_name)

        min_time = min(self.time)

        cons_name = 'storage_balance'
        constr = (self.vars['tank_level'][sto,t]
                  == self.vars['tank_level'][sto,next_t[t]] + sum(self.vars['arc_flow'][a,sto,t] for a in asset_to_storage[sto])
                  - sum(self.vars['arc_flow'][sto,a,t] for a in storage_to_asset[sto])
                  for sto in self.storage for t in next_t.keys())
        self.cons[cons_name] = self.model.addConstrs(constr, name=cons_name)
        
        cons_name = 'initial_storage_balance'
        constr = (self.vars['tank_level'][sto,min_time] 
                  == self.initial_vol[sto] + sum(self.vars['arc_flow'][a,sto,min_time] for a in asset_to_storage[sto])
                  - sum(self.vars['arc_flow'][sto,a,min_time] for a in storage_to_asset[sto])
                  for sto in self.storage)
        self.cons[cons_name] = self.model.addConstrs(constr, name=cons_name)

    def _node_cons(self) -> None:
        asset_to_node = {node:[a for a in self.assets
                                if (a, node) in self.a_a]
                        for node in self.nodes}

        node_to_asset = {node:[a for a in self.assets
                                if (node, a) in self.a_a]
                        for node in self.nodes}

        cons_name = 'node_balance'
        constr = (sum(self.vars['arc_flow'][a,node,t] for a in asset_to_node[node])
                    == sum(self.vars['arc_flow'][node,a,t] for a in node_to_asset[node])
                for node in self.nodes for t in self.time)
        self.cons[cons_name] = self.model.addConstrs(constr, name=cons_name)

    def _conn_cons(self) -> None:
        cons_name = 'arc_cap'
        constr = (self.vars['arc_flow'][node1,node2,t] 
                  <= self.trans_cap[(node1, node2)][t]
                  for (node1,node2) in self.a_a for t in self.time)
        self.cons[cons_name] = self.model.addConstrs(constr, name=cons_name)

        cons_name = 'arc_active'
        constr = (self.vars['arc_flow'][node1,node2,t] 
                  <= self.vars['arc_active'][node1,node2,t] * self.BigM
                for (node1, node2) in self.a_a for t in self.time)
        self.cons[cons_name] = self.model.addConstrs(constr, name=cons_name)

        #for bidirectional arcs


    def build_model(self) -> None:
        self._initialize_gurobi()
        self.create_sets_and_parameters()
        self.create_variables()
        self.create_constraints()

    def plot_network(self) -> None:
        self.InputData.plot_network()

    def create_objective(self) -> None:
        #goal is to maximize overall revenue where overall revenue is defined as follows:
        #revenue paid to company to dispose water from supplier - transport cost - storage cost - inj cost - slack costs - penalty
        revenue = sum(self.vars['source_output'][s,t] * self.src_cost[s][t] for s in self.src for t in self.time)
        transport = sum(self.vars['arc_flow'][node1,node2,t] * self.trans_cost[(node1,node2)][t] for (node1, node2) in self.a_a for t in self.time)
        storage = sum(self.vars['tank_level'][sto,t] * self.storage_cost[sto][t] for sto in self.storage for t in self.time)
        injection = sum(self.vars['sink_input'][d,t] * self.sink_cost[d][t] for d in self.sink for t in self.time)
        sup_slack = sum(self.vars['supply_slack'][s,t] * self.slack_supply_cost for s in self.src for t in self.time)
        dem_slack =  sum(self.vars['demand_slack'][d,t] * self.slack_demand_cost for d in self.sink for t in self.time)
        penalty = 0 #TODO:penalty may be aggregated or daily, look into it

        # overall_revenue = revenue - (transport + storage + injection + sup_slack + dem_slack + penalty)
        overall_revenue = revenue - (transport + storage + injection + penalty)

        self.model.setObjective(overall_revenue, GRB.MAXIMIZE)
        self.model.update()

    def solve_model(self) -> None:
        self.create_objective()
        self.model.write(self.LP_FILE_PATH)
        self.model.write(self.MPS_FILE_PATH)

        if (self.model.NumVars <= 2000) and (self.model.NumConstrs <= 2000):
            #solve model
            self.model.optimize()
            # LOGGER.info(f'Model Status: {self.model.status}')
            if self.model.status == GRB.INFEASIBLE:
                self.model.computeIIS()
                self.model.write(self.ILP_FILE_PATH)
            elif self.model.status == GRB.INF_OR_UNBD:
                self.model.setParam('DualReductions', 0)
                self.model.optimize()
                if self.model.status == GRB.INFEASIBLE:
                    self.model.computeIIS()
                    self.model.write(self.ILP_FILE_PATH)
            else:
                self.objective = self.model.ObjVal

                #write solution
                self.model.write(self.SOL_FILE_PATH)
                # self.extract_results()
            # LOGGER.info("Time elapsed: %.2f seconds" % (time.time() - START_TIME))
        else:
            # LOGGER.info("Model is too large for Gurobipy free licence, switching to CPLEX")
            print('Model is too large for Gurobipy free licence, switching to CPLEX')
            self.use_pulp=True
            self.pulp_var, self.pulp_model = LpProblem.fromMPS(self.MPS_FILE_PATH)
            self.pulp_solver = pl.CPLEX_CMD(options=['mipdisplay=0'])
            self.pulp_model.solve(self.pulp_solver)
            if self.pulp_model.status == 1:
                #write soln
                self.extract_pulp_variables()
                self.extract_results()
            # LOGGER.info("Time elapsed: %.2f seconds" % (time.time() - START_TIME))

    def extract_pulp_variables(self) -> None:
        pass

    def extract_results(self) -> None:
        pass

    def get_constr(self, name):
        return self.cons[name]


    def get_time(self):
        return self.time
    
    def get_arcs(self):
        return self.a_a
    
    def get_src_cost(self):
        return self.src_cost
    
    def get_src_forecast(self):
        return self.src_forecast

    def get_sink_cap(self):
        return self.sink_cap
    
    def get_trans_cost(self):
        return self.trans_cost
    
    def get_trans_cap(self):
        return self.trans_cap
    

def main_small_case_study():
    #create a small self_case_study, super detailed for testing purposes.
    #define all water production sources
    #initialize sources
    pp01 = SupplySource('pp01')
    pp02 = SupplySource('pp02')
    pp03 = SupplySource('ppO3')
    pp04 = SupplySource('pp04')
    pp05 = SupplySource('pp05')
    pp06 = SupplySource('pp06')
    pp07 = SupplySource('pp07')
    pp08 = SupplySource('pp08')
    pp09 = SupplySource('pp09')
    pp10 = SupplySource('pp10')
    pp11 = SupplySource('pp11')
    pp12 = SupplySource('pp12')
    pp13 = SupplySource('pp13')
    pp14 = SupplySource('pp14')
    pp15 = SupplySource('pp15')

    #add supply vol
    pp01._add_constant_supply(19000, '2023-01-01', '2023-02-21')
    pp02._add_constant_supply(34000, '2023-01-01', '2023-02-21')
    pp03._add_constant_supply(27000, '2023-01-01', '2023-02-21')
    pp04._add_constant_supply(13000, '2023-01-01', '2023-02-21')
    pp05._add_constant_supply(10000, '2023-01-01', '2023-02-21')
    pp06._add_constant_supply(29000, '2023-01-01', '2023-02-21')
    pp07._add_constant_supply(29000, '2023-01-01', '2023-02-21')
    pp08._add_constant_supply(21000, '2023-01-01', '2023-02-21')
    pp09._add_constant_supply(19000, '2023-01-01', '2023-02-21')
    pp10._add_constant_supply(23000, '2023-01-01', '2023-02-21')
    pp11._add_constant_supply(10000, '2023-01-01', '2023-02-21')
    pp12._add_constant_supply(12000, '2023-01-01', '2023-02-21')
    pp13._add_constant_supply(10000, '2023-01-01', '2023-02-21')
    pp14._add_constant_supply(15000, '2023-01-01', '2023-02-21')
    pp15._add_constant_supply(8000, '2023-01-01', '2023-02-21')

    #add supply_cost
    pp01._add_cost(5, ctype='cost')
    pp02._add_cost(3, ctype='cost')
    pp03._add_cost(4, ctype='cost')
    pp04._add_cost(5, ctype='cost')
    pp05._add_cost(3, ctype='cost')
    pp06._add_cost(4, ctype='cost')
    pp07._add_cost(4, ctype='cost')
    pp08._add_cost(5, ctype='cost')
    pp09._add_cost(3, ctype='cost')
    pp10._add_cost(4, ctype='cost')
    pp11._add_cost(5, ctype='cost')
    pp12._add_cost(5, ctype='cost')
    pp13._add_cost(4, ctype='cost')
    pp14._add_cost(4, ctype='cost')
    pp15._add_cost(5, ctype='cost')

    #create the source object dict
    sourceObjDict = {}
    sourceObjDict[pp01._get_name()] = pp01
    sourceObjDict[pp02._get_name()] = pp02
    sourceObjDict[pp03._get_name()] = pp03
    sourceObjDict[pp04._get_name()] = pp04
    sourceObjDict[pp05._get_name()] = pp05
    sourceObjDict[pp06._get_name()] = pp06
    sourceObjDict[pp07._get_name()] = pp07
    sourceObjDict[pp08._get_name()] = pp08
    sourceObjDict[pp09._get_name()] = pp09
    sourceObjDict[pp10._get_name()] = pp10
    sourceObjDict[pp11._get_name()] = pp11
    sourceObjDict[pp12._get_name()] = pp12
    sourceObjDict[pp13._get_name()] = pp13
    sourceObjDict[pp14._get_name()] = pp14
    sourceObjDict[pp15._get_name()] = pp15

    #create sink
    cp01 = DemandSink('cp01')
    cp02 = DemandSink('cp02')
    cp03 = DemandSink('cp03')
    cp04 = DemandSink('cp04')
    k01 = DemandSink('k01')
    k02 = DemandSink('k02')
    k03 = DemandSink('k03')

    #add demand forecast/cap
    cp01._add_constant_forecast(337000, '2023-01-02', '2023-01-11')
    cp02._add_constant_forecast(287000, '2023-01-22', '2023-02-04')
    cp03._add_constant_forecast(269000, '2023-01-13', '2023-01-20')
    cp04._add_constant_forecast(282000, '2023-02-05', '2023-02-21')
    k01._add_constant_forecast(100000, '2023-01-01', '2023-02-21')
    k02._add_constant_forecast(150000, '2023-01-01', '2023-02-21')
    k03._add_constant_forecast(200000, '2023-01-01', '2023-02-21')

    #add cost
    cp01._add_cost(0.25)
    cp02._add_cost(0.25)
    cp03._add_cost(0.25)
    cp04._add_cost(0.25)
    k01._add_cost(0.25)
    k02._add_cost(0.25)
    k03._add_cost(0.25)

    #create sink obj dict
    sinkObjDict = {}
    sinkObjDict[cp01._get_name()] = cp01
    sinkObjDict[cp02._get_name()] = cp02
    sinkObjDict[cp03._get_name()] = cp03
    sinkObjDict[cp04._get_name()] = cp04
    sinkObjDict[k01._get_name()] = k01
    sinkObjDict[k02._get_name()] = k02
    sinkObjDict[k03._get_name()] = k03

    #storage
    s01 = Storage('s01', capacity=100000, cost=0, initalvol=0)
    s02 = Storage('s02', capacity=100000, cost=0, initalvol=0)

    #storage dict
    stoObjDict = {}
    stoObjDict[s01.get_name()] = s01
    stoObjDict[s02.get_name()] = s02

    #create connection arcs
    pi1 = TransportConn('pipe1', upstream='pp01', downstream='N01', fixedcost=0.002, capacity=500)
    pi2 = TransportConn('pipe2', upstream='N01', downstream='k01', fixedcost=0.002, capacity=500)
    pi3 = TransportConn('pipe3', upstream='N01', downstream='N02', fixedcost=2, capacity=500)
    pi4 = TransportConn('pipe4', upstream='N02', downstream='N03', fixedcost=2, capacity=500)
    pi5 = TransportConn('pipe5', upstream='N03', downstream='cp01', fixedcost=2, capacity=500)
    pi6 = TransportConn('pipe6', upstream='N03', downstream='N04', fixedcost=2, capacity=500)
    pi7 = TransportConn('pipe7', upstream='N04', downstream='k02', fixedcost=2, capacity=500)
    pi8 = TransportConn('pipe8', upstream='N04', downstream='N06', fixedcost=2, capacity=500)
    pi9 = TransportConn('pipe9', upstream='pp03', downstream='N06', fixedcost=2, capacity=500)
    pi10 = TransportConn('pipe10', upstream='N02', downstream='N05', fixedcost=2, capacity=500)
    pi11 = TransportConn('pipe11', upstream='N06', downstream='N07', fixedcost=2, capacity=500)
    pi12 = TransportConn('pipe12', upstream='pp02', downstream='N05', fixedcost=2, capacity=500)
    pi13 = TransportConn('pipe13', upstream='N05', downstream='N08', fixedcost=2, capacity=500)
    pi14 = TransportConn('pipe14', upstream='N08', downstream='N07', fixedcost=2, capacity=500)
    pi15 = TransportConn('pipe15', upstream='N08', downstream='s01', fixedcost=2, capacity=500)
    pi16 = TransportConn('pipe16', upstream='s01', downstream='cp01', fixedcost=2, capacity=500)
    pi17 = TransportConn('pipe17', upstream='s01', downstream='cp02', fixedcost=2, capacity=500)
    pi18 = TransportConn('pipe18', upstream='N07', downstream='N09', fixedcost=2, capacity=500)
    pi19 = TransportConn('pipe19', upstream='N09', downstream='N10', fixedcost=2, capacity=500)
    pi20 = TransportConn('pipe20', upstream='N10', downstream='cp02', fixedcost=2, capacity=500)
    # pi1 = TransportConn('pipe1', upstream='pp01', downstream='N01', fixedcost=2, capacity=500)
    # pi1 = TransportConn('pipe1', upstream='pp01', downstream='N01', fixedcost=2, capacity=500)
    # pi1 = TransportConn('pipe1', upstream='pp01', downstream='N01', fixedcost=2, capacity=500)


    #pipedict
    connObjDict = {}
    connObjDict[pi1.get_name()] = pi1
    connObjDict[pi2.get_name()] = pi2
    connObjDict[pi3.get_name()] = pi3
    connObjDict[pi4.get_name()] = pi4
    connObjDict[pi5.get_name()] = pi5
    connObjDict[pi6.get_name()] = pi6
    connObjDict[pi7.get_name()] = pi7
    connObjDict[pi8.get_name()] = pi8
    connObjDict[pi9.get_name()] = pi9
    connObjDict[pi10.get_name()] = pi10
    connObjDict[pi11.get_name()] = pi11
    connObjDict[pi12.get_name()] = pi12
    connObjDict[pi13.get_name()] = pi13
    connObjDict[pi14.get_name()] = pi14
    connObjDict[pi15.get_name()] = pi15
    connObjDict[pi16.get_name()] = pi16
    connObjDict[pi17.get_name()] = pi17
    connObjDict[pi18.get_name()] = pi18
    connObjDict[pi19.get_name()] = pi19
    connObjDict[pi20.get_name()] = pi20



    #create
    inp = Mathmodel(name='small_case', optType='b',
                    sourceObjDict=sourceObjDict, 
                    sinkObjDict=sinkObjDict, 
                    stoObjDict=stoObjDict, 
                    connObjDict=connObjDict)
    
    inp.build_model()
    inp.plot_network()
    inp.solve_model()
    # print(inp.get_arcs())




if __name__ == '__main__':
    # #create some test Source Data and save it into a source Object dict
    # #source1: constant supply 500bwpd
    # so1 = SupplySource("prd1", -98.1, 104.2)
    # so1._add_constant_supply(500, '2020-01-01', '2020-02-01')
    # so1._add_cost(50, ctype='cost')
    # #source2: constant supply 1000bwpd
    # so2 = SupplySource('prd2', -99.05, 104.36)
    # so2._add_constant_supply(1000, '2020-01-15', '2020-03-01')
    # so2._add_cost(30, ctype='cost')

    # #create the source object dict
    # sourceObjDict = {}
    # sourceObjDict[so1._get_name()] = so1
    # sourceObjDict[so2._get_name()] = so2
    

    # #create some test demand data and save it into a sink object dict
    # #sink1: 
    # si1 = DemandSink("disp1", -96.2, 106)
    # si1._add_constant_forecast(450, '2020-01-01', '2020-02-15')
    # si1._add_cost(7)

    # #sink2:
    # si2 = DemandSink("disp2", -94.5, 103)
    # si2._add_constant_forecast(800, '2020-01-25', '2020-03-01')
    # si2._add_cost(4)

    # sinkObjDict = {}
    # sinkObjDict[si1._get_name()] = si1
    # sinkObjDict[si2._get_name()] = si2

    # #create test storage
    # #sto1:
    # sto1 = Storage('sto1', -94.5, 101, 10000, cost=3, initalvol=3000)
    # #sto2:
    # sto2 = Storage('sto2', -94.9, 103, 12000, cost=0.5, initalvol=5000)
    
    # stoObjDict = {}
    # stoObjDict[sto1.get_name()] = sto1
    # stoObjDict[sto2.get_name()] = sto2
    

    # #create pipelines
    # #pipe1:
    # pi1 = TransportConn('pipe1', upstream='prd1', downstream='sto1', fixedcost=2, capacity=500)
    # #pipe2:
    # pi2 = TransportConn('pipe2', upstream='sto1', downstream='disp1', fixedcost=3, capacity=100)
    # #pipe3:
    # pi3 = TransportConn('pipe3', upstream='prd2', downstream='disp2', fixedcost=5, capacity=80)
    # #pipe4:
    # pi4 = TransportConn('pipe4', upstream='prd2', downstream='sto2', fixedcost=2.5, capacity=200)
    # #pipe5:
    # pi5 = TransportConn('pipe5', upstream='sto2', downstream='disp2', fixedcost=3.5, capacity=300)
    # #pipe6:
    # pi6 = TransportConn('pipe6', upstream='sto1', downstream='sto2', fixedcost=0.3, capacity=150)
    # #pipe7
    # pi7 = TransportConn('pipe7', upstream='prd1', downstream='node1', fixedcost=0.1)
    # #pipe8
    # pi8 = TransportConn('pipe8', upstream='node1', downstream='sto2', fixedcost=0.1)

    # connObjDict = {}
    # connObjDict[pi1.get_name()] = pi1
    # connObjDict[pi2.get_name()] = pi2
    # connObjDict[pi3.get_name()] = pi3
    # connObjDict[pi4.get_name()] = pi4
    # connObjDict[pi5.get_name()] = pi5
    # connObjDict[pi6.get_name()] = pi6
    # connObjDict[pi7.get_name()] = pi7
    # connObjDict[pi8.get_name()] = pi8


    # #create
    # inp = Mathmodel(name='modela', optType='b',
    #                 sourceObjDict=sourceObjDict, 
    #                 sinkObjDict=sinkObjDict, 
    #                 stoObjDict=stoObjDict, 
    #                 connObjDict=connObjDict)
    
    # inp.build_model()
    # inp.solve_model()
    # print(inp.get_arcs())

    main_small_case_study()
