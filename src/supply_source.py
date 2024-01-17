import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple

class SupplySource:
    def __init__(self, name: str, lat: float = None, lon: float = None):
        self.name = name
        self.lat = lat
        self.lon = lon 
        self.supplycost = 0 #this should be negative if we have revenue from operator not cost
        self.forecastAdded: bool = False
        self.hasTarget: bool = False #if supply has contractual target that must be met
        self.penalty = 0 #penalty for missing contractual target
        self.start_date = None
        self.end_date = None

    
    def _add_supply_forecast(self, forecast: pd.DataFrame) -> None:
        """
            Function embeds source supply forecast per time period

            Parameters
            ---------
            forecast: Pandas Dataframe containing 2 columns:
            col1: Date(YYYY-MM-DD)
            col2: Volume(bbls)


            Return
            ------
            None.
        """
        forecast['Date'] = pd.to_datetime(forecast['Date'])
        self.forecast_dict = dict(zip(forecast.iloc[:, 0], forecast.iloc[:, 1]))
        self.forecastAdded = True
        self.start_date = min(forecast['Date']).strftime('%Y-%m-%d')
        self.end_end = max(forecast['Date']).strftime('%Y-%m-%d')
    
    def _add_constant_supply(self, supplyval: float, start_date: str, end_date: str) -> None:
        """
            Function to add single value forecast over given duration

            Parameters
            ---------
            supplyval: Single value water production or supply in bbls
            start_date: Forecast start Date(YYYY-MM-DD) as string
            end_date: Forecast End Date(YYYY-MM-DD) as string


            Return
            ------
            None.
        """
        #convert data to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        #get date range
        date_range = pd.date_range(start_date, end_date)
        
        #replicate supply val in a list
        supply = [supplyval]*len(date_range)

        self.forecast_dict = dict(zip(date_range, supply))
        self.forecastAdded = True
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.end_date = end_date.strftime('%Y-%m-%d')

    def _add_cost(self, cost: float, ctype='revenue'):
        """
            Function to add cost of water supply or revenue from water supply 
            in $/bbl

            Parameters
            ---------
            cost: positive(cost) or negative(revenue) value in $/bbl
            ctype: indicates if it is a cost or revenue. Default is revenue

            Return
            ------
            None.
        """
        if ctype == 'revenue': #TODO: change this
             self.supplycost = abs(cost) * -1
        elif ctype == 'cost':
            self.supplycost = abs(cost)

    
    def _add_target(self, target_vol, target_dur='1 week', target_frequency='weekly'):

        self.hasTarget = True
        pass #TODO

    def _add_penalty(self, cost_penalty: float) -> None:
        """
            Function to add cost penalty per barrel of supply missed

            Parameters
            ---------
            cost_penalty: cost of missing contratual target in $/bbl


            Return
            ------
            None.
        """
        if self.hasTarget:
            self.penalty = cost_penalty
        else:
            print("please set contractual target before setting penalty")


    
    def _get_all_forecast(self) -> Dict:
        return self.forecast_dict
    
    def _get_supply_vol(self, input_date: str) -> float:
        input_date = pd.to_datetime(input_date)
        return self.forecast_dict[input_date]
    
    def _get_name(self) -> str:
        return self.name
    
    def _get_loc(self) -> Tuple[float, float]:
        return self.lat, self.lon
    
    def _get_cost(self) -> float:
        return self.supplycost
    
    def get_dates(self):
        return self.start_date, self.end_date
    
    def _hasTarget(self) -> bool:
        return self.hasTarget
    
    def _hasforecast(self) -> bool:
        return self.forecastAdded
    
    def _get_penalty(self) -> float:
        return self.penalty
    
    def _plot_forecast(self) -> None:
        x = self.forecast_dict.keys()
        y = self.forecast_dict.values()
        plt.plot(x, y)
        plt.xlabel('Date')
        plt.ylabel('Volume (bbls)')
        plt.title(f"{self.name} supply forecast")
        plt.show()
    

if __name__ == '__main__':
    #testing single value allocation, with the getter and setter methods
    s1 = SupplySource("S1", 104.1, -97.2)

    s1._add_constant_supply(1000, "2020-01-01", "2020-03-31")

    s1._add_cost(5)

    s1._add_target(1000)

    s1._add_penalty(1.5)

    print(s1._get_all_forecast())
    print(s1._get_supply_vol("2020-01-30"))
    print(s1._get_cost())
    lat, long = s1._get_loc()
    print(lat, long)
    print(s1._hasTarget())
    print(s1._get_name())
    print("")

    #Testing saving SupplySource Objects in a dict and calling methods on the dict values
    all_sources = {}

    all_sources[0] = s1
    all_sources[1] = SupplySource("S2", 104.1, -97.2)

    all_sources[1]._add_constant_supply(200, "2021-06-20", "2022-08-28")
    all_sources[1]._add_cost(4.7, ctype="cost")

    for key in all_sources.keys():
        print('key', key)
        print(all_sources[key]._get_name())
        print(all_sources[key]._get_loc())
        print(all_sources[key]._get_cost())
        all_sources[key]._plot_forecast()
        print("")
