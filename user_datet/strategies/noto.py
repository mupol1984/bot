# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame,Series
from typing import Dict, Optional, Union, Tuple , Any
from freqtrade.strategy import (IStrategy,informative,BooleanParameter,CategoricalParameter,DecimalParameter,IntParameter,RealParameter,timeframe_to_minutes,timeframe_to_next_date,timeframe_to_prev_date,merge_informative_pair,stoploss_from_absolute,stoploss_from_open)
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from technical.qtpylib import crossed_above , crossed_below
from freqtrade.persistence import Trade , Order
from technical.qtpylib import crossed_above , crossed_below
from functools import reduce
import math

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series((highest_high - dataframe["close"]) / (highest_high - lowest_low),name=f"{period} Williams %R")
    return WR * -100


class noto(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short:bool = True
    max_open_trades = 1

    minimal_roi = {
        '0': 0.1
    }
    stoploss = -0.9

    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 500


    
    # mama_dif1 = DecimalParameter(0.9 , 1.3 , default= 0.982 , decimals= 3 , space='buy' , optimize= True)
    # mama_dif2 = DecimalParameter(0.9 , 1.3 , default= 1.01 , decimals= 3 , space='sell', optimize= True)

    buy_williams = DecimalParameter(-100 , -30 , default= -60.1 , decimals= 1 , space='buy', optimize= True)
    sell_williams = DecimalParameter(-70 , -5 , default= -40 , decimals= 1 , space='sell', optimize= True)

    # buy_dif = DecimalParameter(-0.3 , 0 , default= -0.025 , decimals= 3 , space='buy', optimize= True)
    # sell_dif = DecimalParameter(0 , 0.3 , default= 0.025 , decimals= 3 , space='sell', optimize= True)

    buy_cti = DecimalParameter(-1.5 , 0 , default= -0.751 , decimals= 3 , space='buy', optimize= True)
    sell_cti = DecimalParameter(0 , 1.5 , default= 0.751 , decimals= 3 , space='sell', optimize= True)

    # buy_rsi = IntParameter(0 , 80 , default= 60 , space= 'buy')
    # sell_rsi = IntParameter(20 , 100 , default= 40 , space= 'sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.25, 0.025)
        dataframe['mama_diff'] = ( ( dataframe['mama'] - dataframe['fama'] ) / dataframe['hl2'] )
        dataframe['kama'] = ta.KAMA(dataframe['close'], 84)

        dataframe['cti'] = pta.cti(dataframe["close"], length=20)   #  -1  to 1

        dataframe['r14'] = williams_r(dataframe, period=14)   # -100  to 0

        dataframe['rsi84'] = ta.RSI(dataframe, timeperiod=84)    # 0  to 100
        # dataframe['rsi112'] = ta.RSI(dataframe, timeperiod=112)

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['kama'] > dataframe['fama']) &
                # (dataframe['fama'] > dataframe['mama'] * self.mama_dif1.value) &
                (dataframe['fama'] > dataframe['mama']) &
                (dataframe['r14'] < self.buy_williams.value) &
                # (dataframe['mama_diff'] < self.buy_dif.value) &
                (dataframe['cti'] < self.buy_cti.value)&
                # (dataframe['close'].rolling(48).max() >= (dataframe['close'] * 1.05 )) &
                # ( (dataframe['close'].rolling(288).max() >= (dataframe['close'] * 1.125 )) )&
                # (dataframe['rsi84'] < self.buy_rsi.value) &
                # (dataframe['rsi_112'] < 60)
                (dataframe['volume'] > 0)),'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['kama'] < dataframe['fama']) &
                # (dataframe['fama'] < dataframe['mama'] * self.mama_dif2.value) &
                (dataframe['fama'] < dataframe['mama']) &
                (dataframe['r14'] > self.sell_williams.value) &
                # (dataframe['mama_diff'] > self.sell_dif.value) &
                (dataframe['cti'] > self.sell_cti.value)&
                # (dataframe['close'].rolling(48).max() >= (dataframe['close'] * 1.05 )) &
                # ( (dataframe['close'].rolling(288).max() >= (dataframe['close'] * 1.125 )) )&
                # (dataframe['rsi84'] > self.sell_rsi.value) &
                # (dataframe['rsi_112'] < 60)
                (dataframe['volume'] > 0)),'enter_short'] = 1
        
        return dataframe
    
    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str,side: str,**kwargs) -> float:
        return 5