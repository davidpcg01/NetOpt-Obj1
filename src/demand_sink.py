import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple

class DemandSink:
    def __init__(self, name: str, lat: float = None, lon: float = None) -> None:
        self.name = name
        self.lat = lat 
        self.lon = lon
        self.forecastAdded = False
        self.demandcost = 0
        self.start_date = None
        self.end_date = None

    def _add_demand_forecast(self, forecast: pd.DataFrame) -> None:
        """
            Function embeds sink demand forecast per time period

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
        self.end_date = max(forecast['Date']).strftime('%Y-%m-%d')

    def _add_constant_forecast(self, demandvol, start_date, end_date) -> None:
        """
            Function to add single value forecast over given duration

            Parameters
            ---------
            supplyval: Single value water injection capacity in bbls
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
        supply = [demandvol]*len(date_range)

        self.forecast_dict = dict(zip(date_range, supply))
        self.forecastAdded = True
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.end_date = end_date.strftime('%Y-%m-%d')

    def _add_cost(self, cost: float) -> None:
        """
            Function to add cost of water injetion or disposal
            in $/bbl

            Parameters
            ---------
            cost: value in $/bbl

            Return
            ------
            None.
        """
        self.demandcost = cost

    def _get_name(self) -> str:
        return self.name
    
    def _get_loc(self) -> Tuple[float, float]:
        return self.lat, self.lon
    
    def _get_all_forecast(self) -> Dict:
        return self.forecast_dict
    
    def _get_demand_vol(self, input_date) -> float:
        input_date = pd.to_datetime(input_date)
        return self.forecast_dict[input_date]
    
    def _get_cost(self) -> float:
        return self.demandcost
    
    def get_dates(self):
        return self.start_date, self.end_date
    
    def _hasforecast(self) -> bool:
        return self.forecastAdded
    
    def _plot_forecast(self) -> None:
        x = self.forecast_dict.keys()
        y = self.forecast_dict.values()
        plt.plot(x, y)
        plt.xlabel('Date')
        plt.ylabel('Volume (bbls)')
        plt.title(f"{self.name} demand forecast")
        plt.show()
    
if __name__ == '__main__':
    d1 = DemandSink("D1", 106.5, -97.68)

    d1._add_constant_forecast(500, "2020-01-01", "2022-01-01")

    d1._add_cost(0.85)

    print(d1._get_name())
    lat, lon = d1._get_loc()
    print(lat, lon)
    print(d1._get_cost())
    print(d1._get_demand_vol("2021-08-02"))
    print(d1.get_dates())
    d1._plot_forecast()