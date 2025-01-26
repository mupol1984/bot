import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from freqtrade.strategy import informative,  IStrategy
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
import math
import numpy as np



def hull(close:pd.Series , length:int)->pd.Series:
    w1 = pta.wma(close=close , length= length)
    w2 = pta.wma(close=close , length= int(length/2))
    w3 = ((w2*2) - w1)
    hull = pta.wma(close=w3 , length=int((math.sqrt(length))))
    return hull




class god3(IStrategy):

    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_short: bool = True
    max_open_trades = 1


    minimal_roi = {
        '0': 0.5}

    stoploss = -0.6

    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    startup_candle_count: int = 800


    order_types = {
        'entry': 'market', 
        'exit': 'market', 
        'stoploss': 'market',       
        'stoploss_on_exchange': False}


    # @property
    # def plot_config(self):
    #     return {
    #         'main_plot': {
    #             'hull': {'color': '#FF8F00'},
    #         },
    #         'subplots': {
    #             "QQE": {
    #                 'qqel': {'color': '#4CAF50'},
    #                 'qqes': {'color': '#F44336'},
    #             },
    #         }
    #     }


    @informative('5m')
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['hull'] = hull(close=dataframe['close'] , length= 80)
        dataframe['cci'] = pta.cci(high=dataframe['high'] , low=dataframe['low'] , close=dataframe['close'] , length= 14)
        dataframe['rsi'] = pta.rsi(close=dataframe['close'] , length= 14)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)        
        
        
        return dataframe



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['tema'] = pta.tema(close=dataframe['close'] , length= 10)
        
        return dataframe



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[((dataframe['tema'] > dataframe['hull_5m']) & 
                      (dataframe['cci_5m'] <-100) & 
                    #   (dataframe['fisher_rsi_5m'] < -0.5) & 
                      (dataframe['rsi_5m'] < 45) &
                      (dataframe['volume'] > 0)), 'enter_long'] = 1
        

        dataframe.loc[((dataframe['tema'] < dataframe['hull_5m']) & 
                      (dataframe['cci_5m']> 100) & 
                    #   (dataframe['fisher_rsi_5m'] > 0.5) & 
                      (dataframe['rsi_5m'] > 55) &
                      (dataframe['volume'] > 0)), 'enter_short'] = 1
        # dataframe.loc[((qtpylib.crossed_above(dataframe['tema'] , dataframe['hull_5m'])) & 
        #             #   (dataframe['cci_5m'] <= -50.0) & 
        #               (dataframe['fisher_rsi_5m'] < -0.5) & 
        #               (dataframe['volume'] > 0)), 'enter_long'] = 1
        

        # dataframe.loc[((qtpylib.crossed_below(dataframe['tema'] , dataframe['hull_5m'])) & 
        #             #   (dataframe['cci_5m'] >= 50.0) & 
        #               (dataframe['fisher_rsi_5m'] > 0.5) & 
        #               (dataframe['volume'] > 0)), 'enter_short'] = 1    

        # dataframe.loc[(qtpylib.crossed_above(dataframe['macd_5m'] ,dataframe['macd_signal_5m'])) & 
        #                 (dataframe['macd_5m'] < -0) &
        #             #   (dataframe['close_5m'] > dataframe['rma_5m']) &
        #                 (dataframe['rsi'] < 50) &
        #               (dataframe['volume'] < dataframe['volume'].rolling(window=30).mean().shift(1) * 20), 'enter_long'] = 1

        # dataframe.loc[(qtpylib.crossed_below(dataframe['macd_5m'] ,dataframe['macd_signal_5m'])) & 
        #                 (dataframe['macd_5m'] > 0) &
        #                 (dataframe['rsi'] > 50) &
        #               (dataframe['volume'] < dataframe['volume'].rolling(window=30).mean().shift(1) * 20), 'enter_short'] = 1
        
        return dataframe




    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[df['enter_long'] == 1, 'exit_short'] = 1
        df.loc[df['enter_short'] == 1, 'exit_long'] = 1

        return df



    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str,side: str,**kwargs) -> float:
        return 5.0


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                                proposed_stake: float, min_stake: Optional[float], max_stake: float,
                                leverage: float, entry_tag: Optional[str], side: str,
                                **kwargs) -> float:

            amount = (max_stake / 2)
            if min_stake < amount < max_stake:
                return amount
            
            if amount < min_stake:
                return min_stake
