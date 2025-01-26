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

def rmi(dataframe:DataFrame , length=20, mom=5):
    df = dataframe.copy()
    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)
    df.fillna(0, inplace=True)
    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)
    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))
    return df["RMI"]

def ewo(dataframe:DataFrame, ema_length=5, ema2_length=35):
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

def vwma(dataframe: DataFrame, length: int = 10):
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma

def sroc(dataframe:DataFrame, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()
    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)
    return sroc

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series((highest_high - dataframe["close"]) / (highest_high - lowest_low),name=f"{period} Williams %R")
    return WR * -100



class met(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short:bool = True
    max_open_trades = 1

    minimal_roi = {
        '0': 0.1
    }
    stoploss = -0.9

    #! ###############################################################
    

    #! ###############################################################

    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 500

    buy_cmf = DecimalParameter(-0.4 , 0.1 , default= -0.12 , decimals= 2 , space='buy' , optimize= True)
    sell_cmf = DecimalParameter(-0.1 , 0.4 , default= 0.12 , decimals= 2 , space='sell' , optimize= True)

    # buy_dif1 = DecimalParameter(1.0 , 1.4 , default= 1.01 , decimals= 3 , space='buy' , optimize= True)
    # sell_dif1 = DecimalParameter(1.0 , 1.4 , default= 1.1 , decimals= 3 , space='sell' , optimize= True)

    # buy_dif2 = DecimalParameter(1.0 , 1.4 , default= 1.01 , decimals= 3 , space='buy' , optimize= True)
    # sell_dif2 = DecimalParameter(1.0 , 1.4 , default= 1.1 , decimals= 3 , space='sell' , optimize= True)
    # buy_sroc = DecimalParameter(-2 , 0 , default= -1.5 , decimals= 2 , space='buy')
    # sell_sroc = DecimalParameter(0 , 2 , default= 1.5 , decimals= 2 , space='sell')


    # buy_adx = IntParameter(10,40 , default= 25 , space='buy', optimize= True)
    # sell_adx = IntParameter(10,40 , default= 25 , space='sell', optimize= True)

    # buy_rsi = IntParameter(0,60 , default= 25 , space='buy')
    # sell_rsi = IntParameter(40,100 , default= 65 , space='sell')

    # buy_r = DecimalParameter(-100 , -40 , default= -50 , decimals= 1 , space='buy', optimize= True)
    # sell_r = DecimalParameter(-60 , 0 , default= 50 , decimals= 1 , space='sell', optimize= True)


    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        df['hl2'] = (df['high'] + df['low']) / 2
        df['tema'] = pta.tema(df['close'] , 6)
        df['ema5'] = pta.ema(df['hl2'] , 5)
        df['ema12'] = pta.ema(df['hl2'] , 12)
        df['ema25'] = pta.ema(df['hl2'] , 25)
        df['ema30'] = pta.ema(df['hl2'] , 30)
        df['ema50'] = pta.ema(df['hl2'] , 50)
        df['ema60'] = pta.ema(df['hl2'] , 60)
        df['ema75'] = pta.ema(df['hl2'] , 75)
        df['ema90'] = pta.ema(df['hl2'] , 90)
        df['ema100'] = pta.ema(df['hl2'] , 100)
        df['ema125'] = pta.ema(df['hl2'] , 125)
        df['ema150'] = pta.ema(df['hl2'] , 150)
        df['ema200'] = pta.ema(df['hl2'] , 200)
        df['cmf'] = chaikin_money_flow(dataframe= df , length= 84)
        df['cti'] = pta.cti(df["close"], length=20)
        df['mama'], df['fama'] = ta.MAMA(df['hl2'], 0.25, 0.025)
        df['kama'] = ta.KAMA(df['close'], 84)
        df['r14'] = williams_r(df, period=14)
        # df['rsi'] = pta.rsi(df['close'] ,14)
        df['sroc'] = sroc(dataframe= df  , roclen= 21 ,emalen= 13 , smooth= 12)
        # df["roc"] = ta.ROC(df)
        # df['adx'] = ta.ADX(df)
        # df['ewo'] = ewo(df , 5 ,50)
        # df['tsi'] = tsi(dataframe= df , slow_len= 26 , fast_len= 6)
        # df['rmi'] = rmi(dataframe= df , length= 20 , mom= 5)

        return df


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        

        df.loc[(
            (df['close'] > df['ema25']) &
            (df['ema25'] < df['ema50']) &
            # (df['ema50'] < df['ema75']) &
            # (df['ema60'] < df['ema90']) &
            # (df['ema50_5m'] < df['ema100_5m']) &
            # (df['ema26'] < df['ema48']) &
            # (df['ema75'] < df['ema100']) &
            # (df['ema75'] < df['ema125']) &
            # (df['ema150'] < df['ema200']) &
            # ((df['close'].rolling(288).max() >= (df['close'] * self.buy_dif2.value )))&
            (df['cmf'] < self.buy_cmf.value) &
            # (df['r14'] < self.buy_r.value) &
            # (df['sroc'] < self.buy_sroc.value) &
            (df['volume'] > 0)), 'enter_long'] = 1
        
        df.loc[(
            (df['close'] < df['ema25']) &
            (df['ema25'] > df['ema50']) &
            # (df['ema50'] > df['ema75']) &
            # (df['ema60'] > df['ema90']) &
            # (df['ema50_5m'] > df['ema100_5m']) &
            # (df['ema75'] > df['ema100']) &
            # (df['ema150'] > df['ema200']) &
            # ((df['close'].rolling(288).min() <= (df['close'] * self.sell_dif2.value )))&
            (df['cmf'] > self.sell_cmf.value) &
            # (df['r14'] > self.sell_r.value) &
            # (df['sroc'] > self.sell_sroc.value) &
            (df['volume'] > 0)), 'enter_short'] = 1
        
        
        return df
    

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[((df['enter_long'] == 1) & (df['volume'] > 0)) , 'exit_short'] = 1
        df.loc[((df['enter_short'] == 1)  & (df['volume'] > 0)) , 'exit_long'] = 1
        return df
    
    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str,side: str,**kwargs) -> float:
        return 5
    

    # def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,current_profit: float, **kwargs):

    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     cn1 = dataframe.iloc[-1].squeeze()
    #     cn2 = dataframe.iloc[-2].squeeze()
    #     if trade.is_open and trade.entry_side == 'buy':
    #         if (cn1['close'] < cn1['ema100']) and (cn2['close'] >= cn2['ema100']):
    #         # if cn1['sroc'] > 0 and cn2['sroc'] < 0:
    #             return 'exit_long_cross'
        
    #     if trade.is_open and trade.entry_side =='sell':
    #         if (cn1['close'] > cn1['ema100']) and (cn2['close'] <= cn2['ema100']):
    #         # if cn1['sroc'] < 0 and cn2['sroc'] > 0:
    #             return 'exit_short_cross'
        
    #     return None
    

    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
    #                         proposed_stake: float, min_stake: float, max_stake: float,
    #                         **kwargs) -> float:
    #     if (self.config['position_adjustment_enable'] == True) and (self.config['stake_amount'] == 'unlimited'):
    #         return self.wallets.get_total_stake_amount() / self.config['max_open_trades'] / self.max_dca_multiplier
    #     else:
    #         return proposed_stake
