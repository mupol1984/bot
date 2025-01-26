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

def ewo(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["close"] * 100
    return emadif

def chaikin_money_flow(dataframe:DataFrame, length=20, fillna=False) -> Series:
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(length, min_periods=0).sum() / df['volume'].rolling(length, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def rmi(dataframe:DataFrame , length=20, mom=5):
    df = dataframe.copy()
    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)
    df.fillna(0, inplace=True)
    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)
    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))
    return df["RMI"]

def zema(dataframe:DataFrame, period, field='close'):
    df = dataframe.copy()
    df['ema1'] = ta.EMA(df[field], timeperiod=period)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=period)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']
    return df['zema']

def sroc(dataframe:DataFrame, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()
    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)
    return sroc

def vwma(dataframe: DataFrame, length: int = 10):
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma

def tsi(dataframe: DataFrame, slow_len: int, fast_len: int, fillna=False) -> Series:
    df = dataframe.copy()
    min_periods_slow = 0 if fillna else slow_len
    min_periods_fast = 0 if fillna else fast_len
    close_diff            = df['close'].diff()
    close_diff_abs        = close_diff.abs()
    smooth_close_diff     = close_diff.ewm(span=slow_len, min_periods=min_periods_slow, adjust=False).mean().ewm(span=fast_len, min_periods=min_periods_fast, adjust=False).mean()
    smooth_close_diff_abs = close_diff_abs.ewm(span=slow_len, min_periods=min_periods_slow, adjust=False).mean().ewm(span=fast_len, min_periods=min_periods_fast, adjust=False).mean()
    tsi = smooth_close_diff / smooth_close_diff_abs * 100
    if fillna:
        tsi = tsi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return tsi


class hosna(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short:bool = True
    max_open_trades = 1
    minimal_roi = {
        '0': 0.2
    }

    stoploss = -0.35
    #################################################
    # Buy hyperspace params:
    buy_params = {
        "buy_cmf": -0.596,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_cmf": 0.466,
    }
    ################################################

    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 100

    buy_cmf = DecimalParameter(-0.6 , -0.1 , default= -0.4 , decimals= 3 , space='buy' , optimize= True)
    sell_cmf = DecimalParameter(0.1 , 0.6 , default= 0.25 , decimals= 3 , space='sell' , optimize= True)


    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        df['ema5'] = pta.ema(df['close'] , 5)
        df['ema10'] = pta.ema(df['close'] , 10)
        df['ema20'] = pta.ema(df['close'] , 20)
        df['ema30'] = pta.ema(df['close'] , 30)
        df['ema40'] = pta.ema(df['close'] , 40)
        df['ema50'] = pta.ema(df['close'] , 50)
        # df['ema60'] = pta.ema(df['close'] , 60)
        # df['ema70'] = pta.ema(df['close'] , 70)
        # df['ema80'] = pta.ema(df['close'] , 80)
        # df['ema90'] = pta.ema(df['close'] , 90)
        df['ema100'] = pta.ema(df['close'] , 100)
        # df['ema150'] = pta.ema(df['close'] , 150)
        # df['ema200'] = pta.ema(df['close'] , 200)
        # df['rmi'] = rmi(dataframe= df , length= 20 , mom= 5)
        # df['sroc'] = sroc(dataframe= df , roclen= 21 , emalen= 13 , smooth= 21)
        # df['ewo'] = ewo(df , 5 ,35)
        # df['tsi'] = tsi(df , slow_len= 26 , fast_len= 6)
        # df["mfi"] = ta.MFI(df)
        # df["roc"] = ta.ROC(df)
        df['cmf'] = chaikin_money_flow(dataframe= df , length= 14)

        return df


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[(
            (df['close'] < df['ema5']) &
            (df['ema5'] < df['ema10']) &
            (df['ema10'] < df['ema20']) &
            (df['ema20'] < df['ema30']) &
            (df['ema30'] < df['ema40']) &
            (df['cmf'] < self.buy_cmf.value) &
            (df['volume'] > 0)), 'enter_long'] = 1
        

        df.loc[(
            (df['close'] > df['ema5']) &
            (df['ema5'] > df['ema10']) &
            (df['ema10'] > df['ema20']) &
            (df['ema20'] > df['ema30']) &
            (df['ema30'] > df['ema40']) &
            (df['cmf'] > self.sell_cmf.value) &
            (df['volume'] > 0)), 'enter_short'] = 1
      
        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        df.loc[((df['enter_long'] == 1)) , 'exit_short'] = 1
        df.loc[(df['enter_short'] == 1) , 'exit_long'] = 1
        
        return df


    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str,side: str,**kwargs) -> float:
        
        return 5
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        cn1 = dataframe.iloc[-1].squeeze()
        cn2 = dataframe.iloc[-2].squeeze()
        if trade.is_open and trade.entry_side == 'buy':
            if (cn1['close'] < cn1['ema100']) and (cn2['close'] > cn2['ema100']):
                return 'exit_long_cross'
        
        if trade.is_open and trade.entry_side =='sell':
            if (cn1['close'] > cn1['ema40']) and (cn2['close'] < cn2['ema40']):
                return 'exit_short_cross'
        
        return None


    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,proposed_stake: float, min_stake: float | None, max_stake: float,leverage: float, entry_tag: str | None, side: str,**kwargs) -> float:
    #     stake = self.wallets.get_total_stake_amount() / self.max_open_trades
    #     if stake < min_stake:
    #         return min_stake
        
    #     if stake > max_stake:
    #         return max_stake
        
    #     if min_stake<stake < max_stake:
    #         return stake


# #!    CUSTOM SOPLOSS
#     pHSL = DecimalParameter(-0.990, -0.040, default=-0.08, decimals=3, space='sell' , optimize= True)
#     pPF_1 = DecimalParameter(0.008, 0.100, default=0.016, decimals=3, space='sell' , optimize= True)
#     pSL_1 = DecimalParameter(0.008, 0.100, default=0.011, decimals=3, space='sell' , optimize= True)
#     pPF_2 = DecimalParameter(0.040, 0.200, default=0.080, decimals=3, space='sell' , optimize= True)
#     pSL_2 = DecimalParameter(0.040, 0.200, default=0.040, decimals=3, space='sell' , optimize= True)
#     use_custom_stoploss = True
#     def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,current_rate: float, current_profit: float, **kwargs) -> float:
#         HSL = self.pHSL.value
#         PF_1 = self.pPF_1.value
#         SL_1 = self.pSL_1.value
#         PF_2 = self.pPF_2.value
#         SL_2 = self.pSL_2.value
#         if current_profit > PF_2:
#             sl_profit = SL_2 + (current_profit - PF_2)
#         elif current_profit > PF_1:
#             sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
#         else:
#             sl_profit = HSL
#         if self.can_short:
#             if (-1 + ((1 - sl_profit) / (1 - current_profit))) <= 0:
#                 return 1
#         else:
#             if (1 - ((1 + sl_profit) / (1 + current_profit))) <= 0:
#                 return 1
#         return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)