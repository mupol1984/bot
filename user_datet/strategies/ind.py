# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame , Series
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
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

def zema(dataframe:DataFrame, period, field='close'):
    df = dataframe.copy()
    df['ema1'] = ta.EMA(df[field], timeperiod=period)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=period)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']
    return df['zema']

def pcc(dataframe: DataFrame, period: int = 20, mult: int = 2):
    df = dataframe.copy()
    df['previous_close'] = df['close'].shift()
    df['close_change'] = (df['close'] - df['previous_close']) / df['previous_close'] * 100
    df['high_change'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_change'] = (df['low'] - df['close']) / df['close'] * 100
    df['delta'] = df['high_change'] - df['low_change']
    mid = zema(df, period, 'close_change')
    rangema = zema(df, period, 'delta')
    upper = mid + rangema * mult
    lower = mid - rangema * mult
    return upper, rangema, lower

def ssl_channel_atr(dataframe:DataFrame, length=7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['ssl_Down'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['ssl_Up'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['ssl_Down'], df['ssl_Up']

def sroc(dataframe:DataFrame, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()
    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)
    return sroc

def chaikin_money_flow(dataframe:DataFrame, length=20, fillna=False) -> Series:
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(length, min_periods=0).sum() / df['volume'].rolling(length, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')


def ssl_channel(dataframe:DataFrame, length=7):
    df = dataframe.copy()
    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN))
    df["hlv"] = df["hlv"].ffill()
    df["ssl_Down"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["ssl_Up"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["ssl_Down"], df["ssl_Up"]


def ewo(dataframe:DataFrame, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["close"] * 100
    return emadif


def wavetrend(dataframe:DataFrame, chlen:int= 10, avg:int= 21, smalen:int= 4) ->Series:
    df = dataframe.copy()
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['esa'] = ta.EMA(df['hlc3'], timeperiod=chlen)
    df['d'] = ta.EMA((df['hlc3'] - df['esa']).abs(), timeperiod=chlen)
    df['ci'] = (df['hlc3'] - df['esa']) / (0.015 * df['d'])
    df['tci'] = ta.EMA(df['ci'], timeperiod=avg)
    df['wt1'] = df['tci']
    df['wt2'] = ta.SMA(df['wt1'], timeperiod=smalen)
    df['wt1-wt2'] = df['wt1'] - df['wt2']
    return df['wt1'], df['wt2']

def t3(dataframe:DataFrame, length:int= 5) -> Series:
    df = dataframe.copy()
    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b*b*b
    c2 = 3*b*b+3*b*b*b
    c3 = -6*b*b-3*b-3*b*b*b
    c4 = 1+3*b+b*b*b+3*b*b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']
    return df['T3Average']


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

def hull(close:pd.Series , length:int)->pd.Series:
    w1 = pta.wma(close=close , length= length)
    w2 = pta.wma(close=close , length= int(length/2))
    w3 = ((w2*2) - w1)
    hull = pta.wma(close=w3 , length=int((math.sqrt(length))))
    return hull

def adx(dataframe:DataFrame , length:int = 20) -> Series:
    df = dataframe.copy()
    adx = pta.adx(high=df['high'] , low=df['low'] , close=df['close'] , length=14)
    df['adx'] = adx[adx.columns[0]]
    df['d_plus'] = adx[adx.columns[1]]
    df['d_minus'] = adx[adx.columns[2]]

    return df['adx'] , df['d_plus'] , df['d_minus']

def trend(dataframe:DataFrame , length:int = 10 , multiplier:float = 3):
    df = dataframe.copy()
    t = pta.supertrend(high=df['high'] , low=df['low'] , close= df['close'] , length=length , multiplier= multiplier)
    df['trend'] = t[t.columns[0]]
    df['dir'] = t[t.columns[1]]
    df['long'] = t[t.columns[2]]
    df['short'] = t[t.columns[3]]
    return df['trend'] , df['dir'] , df['long'] , df['short']

def stoch(dataframe:DataFrame , k_len:int = 14 , d_len:int = 3) -> Series:
    df = dataframe.copy()
    st = pta.stoch(high=df['high'] , low= df['low'] , close= df['close'] , k= k_len , d= d_len )
    df['stoch_k'] = st[st.columns[0]]
    df['stoch_d'] = st[st.columns[1]]
    return df['stoch_k'] , df['stoch_d']


def rsi_stoch(dataframe:DataFrame ,length:int= 14 , rsi_len:int = 13 , k_len:int = 14 , d_len:int = 3):
    df = dataframe.copy()
    st = pta.stochrsi(close= df['close']  ,length= length , rsi_length= rsi_len , k= k_len , d= d_len)
    df['rsi_k'] = st[st.columns[0]]
    df['rsi_d'] = st[st.columns[1]]
    return df['rsi_k'] , df['rsi_d']


def heiken(dataframe:DataFrame):
    df = dataframe.copy()
    heikinashi = qtpylib.heikinashi(df)
    df['ha_open'] = heikinashi['open']
    df['ha_close'] = heikinashi['close']  
    return df['ha_open'] , df['ha_close']

def vwma(dataframe: DataFrame, length: int = 10):
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series((highest_high - dataframe["close"]) / (highest_high - lowest_low),name=f"{period} Williams %R")
    return WR * -100


class ind(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short: bool = True

    minimal_roi = {
        "0": 0.04
    }

    stoploss = -0.9

    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 500


    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:

        df['ema5'] = pta.ema(df['close'] , 5)
        df['ema10'] = pta.ema(df['close'] , 10)
        df['ema20'] = pta.ema(df['close'] , 20)
        df['ema30'] = pta.ema(df['close'] , 30)
        df['ema40'] = pta.ema(df['close'] , 40)
        df['ema50'] = pta.ema(df['close'] , 50)
        df['ema100'] = pta.ema(df['close'] , 100)
        df['ema150'] = pta.ema(df['close'] , 150)
        df['ema200'] = pta.ema(df['close'] , 200)

        df['cmf'] = chaikin_money_flow(dataframe= df , length= 48)
        df['sroc'] = sroc(dataframe= df , roclen= 21 , emalen= 13 , smooth= 21)
        df['ewo'] = ewo(df , 5 ,35)
        df['tsi'] = tsi(df , slow_len= 26 , fast_len= 6)
        df["mfi"] = ta.MFI(df)
        df["roc"] = ta.ROC(df)
        df['sar'] = ta.SAR(df)
        df['rmi'] = rmi(df)

        df['hull10'] = hull(close=df['close'] , length= 10)
        df['hull25'] = hull(close=df['close'] , length= 25)
        df['hull50'] = hull(close=df['close'] , length= 50)
        df['hull100'] = hull(close=df['close'] , length= 100)
        df['hull200'] = hull(close=df['close'] , length= 200)

        df['cti'] = pta.cti(df["close"], length=20)
        df['r14'] = williams_r(df, period=14)

        df['upper'] , df['range'] , df['lower'] = pcc(dataframe= df , period= 20 , mult= 2)

        df["ssl_Down"], df["ssl_Up"] = ssl_channel(dataframe= df , length= 7)

        df['wt1'], df['wt2'] = wavetrend(dataframe= df , chlen=10 , avg= 21 , smalen= 4)

        df['t3'] = t3(dataframe= df , length= 5)

        df['trend'] , df['dir'] , df['long'] , df['short'] = trend(dataframe= df , length= 10 , multiplier=3)

        df['stoch_k'] , df['stoch_d'] = stoch(dataframe= df , k_len= 14 , d_len= 3)



        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['volume'] > 0)
            ),
            'enter_long'] = 1
        
        df.loc[
            (
                (df['volume'] > 0)), 'enter_short'] = 1
        # df.loc[
        #     (
        #         (df['kama-3'] > df['kama-21']) & 
        #         (df['macd'] > df['macdsignal']) & 
        #         (df['macd'] > 0.1) & 
        #         (df['macdhist'] > 0.1) &
        #         (df['rmi'] > df['rmi'].shift()) &
        #         (df['rmi'] > )
        #         (df['volume'] < (df['volume_ma'] * 20))),"enter_long"] = 1


        # df.loc[
        #     (
        #         (df["tema"] < df["tema"].shift(1)) &  # Guard: tema is raising
        #         (df["volume"] > 0)  # Make sure Volume is not 0
        #     ),
        #     "enter_short"] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
     
        df.loc[((df['enter_long'] == 1)) , 'exit_short'] = 1
        df.loc[(df['enter_short'] == 1) , 'exit_long'] = 1
        
        return df


    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str,side: str,**kwargs) -> float:
        
        return 5.0