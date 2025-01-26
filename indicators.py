import pandas as pd
from pandas import Series , DataFrame
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.qtpylib import crossed_above , crossed_below
import math
import talib.abstract as ta
import pandas_ta as pta

def same_length(bigger, shorter):
    return np.concatenate((np.full((bigger.shape[0] - shorter.shape[0]), np.nan), shorter))


def ZEMA(dataframe:DataFrame, length:int = 14, field='close'):
    df = dataframe.copy()
    df['ema1'] = ta.EMA(df[field], timeperiod=length)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=length)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']
    return df['zema']


def RMI(dataframe:DataFrame,length:int=20, mom:int=5)-> Series:

    df = dataframe.copy()
    df["maxup"] = (df["close"] - df["close"].shift(mom)).clip(lower=0).fillna(0)
    df["maxdown"] = (df["close"].shift(mom) - df["close"]).clip(lower=0).fillna(0)

    df["emaInc"] = ta.EMA(df, price="maxup", timeperiod=length)
    df["emaDec"] = ta.EMA(df, price="maxdown", timeperiod=length)

    df["RMI"] = np.where(df["emaDec"] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))
    return df["RMI"]



def MASTREAK(dataframe: DataFrame, length: int = 4, field='close') -> Series:

    df = dataframe.copy()
    avgval = ZEMA(df, length, field)
    arr = np.diff(avgval)
    pos = np.clip(arr, 0, 1).astype(bool).cumsum()
    neg = np.clip(arr, -1, 0).astype(bool).cumsum()
    streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)), -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))
    res = same_length(df['close'], streak)

    return res


def PCC(dataframe: DataFrame, length: int = 20, mult: int = 2)->Series:
    df = dataframe.copy()
    df['previous_close'] = df['close'].shift()
    df['close_change'] = (df['close'] - df['previous_close']) / df['previous_close'] * 100
    df['high_change'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_change'] = (df['low'] - df['close']) / df['close'] * 100
    df['delta'] = df['high_change'] - df['low_change']
    mid = ZEMA(df, length, 'close_change')
    rangema = ZEMA(df, length, 'delta')
    upper = mid + rangema * mult
    lower = mid - rangema * mult

    return upper, lower


def SSLChannels(dataframe:DataFrame, length:int=10) ->Series:
    df = dataframe.copy()
    df['smaHigh'] = df['high'].rolling(length).mean()
    df['smaLow'] = df['low'].rolling(length).mean()
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1,np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']



def SSLChannels_ATR(dataframe:DataFrame, length:int= 7) ->Series:
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']



def WAVETREND(dataframe:DataFrame, chlen:int= 10, avg:int= 21, sma_len:int= 4) ->Series:
    df = dataframe.copy()
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['esa'] = ta.EMA(df['hlc3'], timeperiod=chlen)
    df['d'] = ta.EMA((df['hlc3'] - df['esa']).abs(), timeperiod=chlen)
    df['ci'] = (df['hlc3'] - df['esa']) / (0.015 * df['d'])
    df['tci'] = ta.EMA(df['ci'], timeperiod=avg)
    df['wt1'] = df['tci']
    df['wt2'] = ta.SMA(df['wt1'], timeperiod=sma_len)
    df['wt1-wt2'] = df['wt1'] - df['wt2']
    return df['wt1'], df['wt2']

def T3(dataframe:DataFrame, length:int= 5) -> Series:
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
    df['T3'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']
    return df['T3']

#! important
def SROC(dataframe:DataFrame, roc_len:int= 21, ema_len:int= 13, smooth:int= 21) ->Series:
    df = dataframe.copy()
    df['roc'] = ta.ROC(df, timeperiod=roc_len)
    df['ema'] = ta.EMA(df, timeperiod=ema_len)
    sroc = ta.ROC(df['ema'], timeperiod=smooth)
    return sroc


def ADX(dataframe:DataFrame , length:int = 20) -> Series:
    df = dataframe.copy()
    adx = pta.adx(high=df['high'] , low=df['low'] , close=df['close'] , length=14)
    df['adx'] = adx[adx.columns[0]]
    df['d_plus'] = adx[adx.columns[1]]
    df['d_minus'] = adx[adx.columns[2]]

    return df['adx'] , df['d_plus'] , df['d_minus']


def BBANDS(dataframe:DataFrame , length:int = 20 , std:float = 2.0)-> Series:
    df = dataframe.copy()
    b = pta.bbands(close= df['close'] , length= length , std= std)
    df['lower'] = b[b.columns[0]]
    df['mid'] = b[b.columns[1]]
    df['upper'] = b[b.columns[2]]
    return df['upper'] , df['mid'] , df['lower']


def QQE(dataframe:DataFrame , length:int = 20 , smooth:int = 5 , factor:float = 4.7):
    df = dataframe.copy()
    q = pta.qqe(close= df['close'] , length= length , smooth= smooth , factor= factor)
    df['qqe'] = q[q.columns[0]]
    df['qqel'] = q[q.columns[2]]
    df['qqes'] = q[q.columns[3]]
    return df['qqe'] , df['qqel'] , df['qqes']


def TREND(dataframe:DataFrame , length:int = 10 , multiplier:float = 3):
    df = dataframe.copy()
    t = pta.supertrend(high=df['high'] , low=df['low'] , close= df['close'] , length=length , multiplier= multiplier)
    df['trend'] = t[t.columns[0]]
    df['dir'] = t[t.columns[1]]
    df['long'] = t[t.columns[2]]
    df['short'] = t[t.columns[3]]
    return df['trend'] , df['dir'] , df['long'] , df['short']


def STOCH(dataframe:DataFrame , k_len:int = 14 , d_len:int = 3) -> Series:
    df = dataframe.copy()
    st = pta.stoch(high=df['high'] , low= df['low'] , close= df['close'] , k= k_len , d= d_len )
    df['stoch_k'] = st[st.columns[0]]
    df['stoch_d'] = st[st.columns[1]]
    return df['stoch_k'] , df['stoch_d']


def RSI_STOCH(dataframe:DataFrame ,length:int= 14 , rsi_len:int = 13 , k_len:int = 14 , d_len:int = 3):
    df = dataframe.copy()
    st = pta.stochrsi(close= df['close']  ,length= length , rsi_length= rsi_len , k= k_len , d= d_len)
    df['rsi_k'] = st[st.columns[0]]
    df['rsi_d'] = st[st.columns[1]]
    return df['rsi_k'] , df['rsi_d']


def HEIKEN(dataframe:DataFrame):
    df = dataframe.copy()
    heikinashi = qtpylib.heikinashi(df)
    df['ha_open'] = heikinashi['open']
    df['ha_close'] = heikinashi['close']  
    return df['ha_open'] , df['ha_close']


def CCI(dataframe:DataFrame , length:int = 14) -> Series:
    df = dataframe.copy()
    df['cci'] = pta.cci(high=df['high'] , low=df['low'] , close= df['close'] , length= length)
    return df['cci']


def HULL(dataframe:DataFrame , length:int = 20) -> Series:
    df = dataframe.copy()
    w1 = pta.wma(close=df['close'] , length= length)
    w2 = pta.wma(close=df['close'] , length= int(length/2))
    w3 = ((w2*2) - w1)
    df['hull'] = pta.wma(close=w3 , length=int((math.sqrt(length))))
    return df['hull']


def VWAPB(dataframe:DataFrame, length:int=20, std:float= 1) -> Series:
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=length)
    rolling_std = df['vwap'].rolling(window=length).std()
    df['v_lower'] = df['vwap'] - (rolling_std * std)
    df['v_upper'] = df['vwap'] + (rolling_std * std)
    return df['v_lower'], df['vwap'], df['v_upper']

def EWO(dataframe:DataFrame , ema1_len:int = 5 , ema2_len:int = 35)-> Series:
    df = dataframe.copy()
    df['ema1'] = pta.ema(df['close'] , ema1_len)
    df['ema2'] = pta.ema(df['close'] , ema2_len)
    df['ewo'] = ((df['ema1'] - df['ema2']) / df['close']) * 100
    return df['ewo']


def ChikenMoneyFlow(dataframe:DataFrame, length:int= 20, fillna=False) ->Series:
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(length, min_periods=0).sum() / df['volume'].rolling(length, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')


def WILLIAMS(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series((highest_high - dataframe["close"]) / (highest_high - lowest_low), name="{0} Williams %R".format(period))
    return WR * -100


def TSI(dataframe: DataFrame, slow_len: int = 26, fast_len: int = 12, fillna=False) -> Series:
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

def ZELMA(dataframe:DataFrame, length:int = 14):
    lag =  int(math.floor((length - 1) / 2) )
    if isinstance(dataframe, Series):
        ema_data = dataframe  + (dataframe  - dataframe.shift(lag))
    else:
        ema_data = dataframe['close']  + (dataframe['close']  - dataframe['close'] .shift(lag))
    return ta.EMA(ema_data, timeperiod = length)



def ZLHULL(dataframe:DataFrame, length:int = 20):
    lag =  int(math.floor((length - 1) / 2) )
    if isinstance(dataframe, Series):
        wma_data = dataframe + (dataframe  - dataframe.shift(lag))
    else:
        wma_data = dataframe['close'] + (dataframe['close']  - dataframe['close'] .shift(lag))
    return  ta.WMA(2 * ta.WMA(wma_data, int(math.floor(length/2))) - ta.WMA(wma_data, length), int(round(np.sqrt(length))))



def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
    if method == 'HL':
        return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe[
            'low'].rolling(length).min()
    elif method == 'OC':
        return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe[
            'close'].rolling(length).min()
    else:
        raise ValueError(f"Method {method} not defined!")



def MODERI(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(VWMA(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend


def VWMA(dataframe: DataFrame, length: int = 10):
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma


