import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from pandas import Series , DataFrame
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from datetime import datetime, timedelta
from freqtrade.strategy import (IStrategy,informative,
    BooleanParameter,CategoricalParameter,DecimalParameter,IntParameter,RealParameter,timeframe_to_minutes,timeframe_to_next_date,timeframe_to_prev_date,merge_informative_pair,stoploss_from_absolute,stoploss_from_open)

from functools import reduce
from typing import List, Dict, Any
import logging
# -------------------------------------------------------------------------------------------------
# --- logger for parameter merging output, only remove if you remove it further down too! ---------
logger = logging.getLogger(__name__)


def RMI(dataframe, length=20, mom=5):
    df = dataframe.copy()
    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)
    df.fillna(0, inplace=True)
    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)
    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))
    return df["RMI"]

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
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

def EWO(dataframe:DataFrame , ema1_len:int = 5 , ema2_len:int = 35)-> Series:
    df = dataframe.copy()
    df['ema1'] = pta.ema(df['close'] , ema1_len)
    df['ema2'] = pta.ema(df['close'] , ema2_len)
    df['ewo'] = ((df['ema1'] - df['ema2']) / df['close']) * 100
    return df['ewo']

def WaveTrend(dataframe:DataFrame, chlen:int= 10, avg:int= 21, smalen:int= 4) ->Series:
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
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']
    return df['T3Average']

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series((highest_high - dataframe["close"]) / (highest_high - lowest_low),name=f"{period} Williams %R")
    return WR * -100

def TSI(dataframe: DataFrame, slow_len: int, fast_len: int, fillna=False) -> Series:
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

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

def SSLChannels(dataframe, length=7):
    df = dataframe.copy()
    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(
        df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN)
    )
    df["hlv"] = df["hlv"].ffill()
    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["sslDown"], df["sslUp"]

class pol(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    max_open_trades = 1
    can_short:bool = True

    #################################################


    

    ################################################

    minimal_roi = {
        "0": 0.9
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

    buy_dip_threshold_1 = DecimalParameter(0.08, 0.2, default=0.12, space="buy", decimals=2)
    buy_dip_threshold_2 = DecimalParameter(0.02, 0.4, default=0.28, space="buy", decimals=2)

    sell_dip_threshold_1 = DecimalParameter(0.08, 0.2, default=0.12, space="sell", decimals=2)
    sell_dip_threshold_2 = DecimalParameter(0.02, 0.4, default=0.28, space="sell", decimals=2)

    buy_bb40_bbdelta_close = DecimalParameter(0.005, 0.04, default=0.031, space="buy", decimals=3)
    buy_bb40_closedelta_close = DecimalParameter(0.01, 0.03, default=0.021, space="buy", decimals=3)
    buy_bb40_tail_bbdelta = DecimalParameter(0.2, 0.4, default=0.264, space="buy", decimals=3)

    sell_bb40_bbdelta_close = DecimalParameter(0.005, 0.04, default=0.031, space="sell", decimals=3)
    sell_bb40_closedelta_close = DecimalParameter(0.01, 0.03, default=0.021, space="sell", decimals=3)
    sell_bb40_tail_bbdelta = DecimalParameter(0.2, 0.4, default=0.264, space="sell", decimals=3)

    # @informative('1h')
    # def populate_indicators_1h(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:

    #     df["ema50"] = ta.EMA(df, timeperiod=50)
    #     df["ema100"] = ta.EMA(df, timeperiod=100)
    #     df["ema200"] = ta.EMA(df, timeperiod=200)

    #     df["sma200"] = ta.SMA(df, timeperiod=200)
    #     df["sma200_dec"] = df["sma200"] < df["sma200"].shift(20)
    #     # RSI
    #     df["rsi"] = ta.RSI(df, timeperiod=14)

    #     # SSL Channels
    #     ssl_down_1h, ssl_up_1h = SSLChannels(df, 20)
    #     df["ssl_down"] = ssl_down_1h
    #     df["ssl_up"] = ssl_up_1h
    #     df["ssl-dir"] = np.where(ssl_up_1h > ssl_down_1h, "up", "down")

    #     return df



    def populate_indicators(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        # bb_40 = qtpylib.bollinger_bands(df["close"], window=40, stds=2)
        # df["lower"] = bb_40["lower"]
        # df["mid"] = bb_40["mid"]
        # df["bbdelta"] = (bb_40["mid"] - df["lower"]).abs()
        # df["closedelta"] = (df["close"] - df["close"].shift()).abs()
        # df["tail"] = (df["close"] - df["low"]).abs()
        # strategy ClucMay72018
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
        # df["bb_lower"] = bollinger["lower"]
        # df["bb_mid"] = bollinger["mid"]
        # df["bb_upper"] = bollinger["upper"]

        # df["volume_mean_slow"] = df["volume"].rolling(window=30).mean()

        # df["ema12"] = ta.EMA(df, timeperiod=12)
        # df["ema26"] = ta.EMA(df, timeperiod=26)
        # df["ema50"] = ta.EMA(df, timeperiod=50)
        # df["ema200"] = ta.EMA(df, timeperiod=200)

        # df["sma5"] = ta.EMA(df, timeperiod=5)
        # df["sma200"] = ta.SMA(df, timeperiod=200)
        # df["sma200_dec"] = df["sma200"] < df["sma200"].shift(20)
        # df["rsi"] = ta.RSI(df, timeperiod=14)
        # df["mfi"] = ta.MFI(df, timeperiod=14)
        # df["atr"] = ta.ATR(df, timeperiod=14)

        # df["EWO"] = EWO(df, 25, 50)

        df['hl2'] = (df['high'] + df['low']) / 2
        df['mama'], df['fama'] = ta.MAMA(df['hl2'], 0.25, 0.025)
        df['mama_diff'] = ( ( df['mama'] - df['fama'] ) / df['hl2'] )
        df['kama'] = ta.KAMA(df['close'], 84)
        df['cti'] = pta.cti(df["close"], length=20)
        df['r_14'] = williams_r(df, period=14)
        df['rsi_84'] = ta.RSI(df, timeperiod=84)
        df['rsi_112'] = ta.RSI(df, timeperiod=112)
        return df


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        df.loc[
            (
                (df['kama'] > df['fama']) &
                (df['fama'] > df['mama'] * 0.981) &
                (df['r_14'] < -61.3) &
                (df['mama_diff'] < -0.025) &
                (df['cti'] < -0.715)&
                (df['close'].rolling(48).max() >= (df['close'] * 1.05 )) &
                ( (df['close'].rolling(288).max() >= (df['close'] * 1.125 )) )&
                (df['rsi_84'] < 60) &
                (df['rsi_112'] < 60)),'enter_long'] = 1


        df.loc[
            (
                (df['kama'] > df['fama']) &
                (df['fama'] > df['mama'] * 0.981) &
                (df['r_14'] < -61.3) &
                (df['mama_diff'] < -0.025) &
                (df['cti'] < -0.715)&
                (df['close'].rolling(48).max() >= (df['close'] * 1.05 )) &
                ( (df['close'].rolling(288).max() >= (df['close'] * 1.125 )) )&
                (df['rsi_84'] < 60) &
                (df['rsi_112'] < 60)),'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
     
        df.loc[(df['volume'] > 0) , 'exit_long'] = 0
        df.loc[(df['volume'] > 0) , 'exit_short'] = 0
        return df

    # def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,current_profit: float, **kwargs):

    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     cn1 = dataframe.iloc[-1].squeeze()
    #     cn2 = dataframe.iloc[-2].squeeze()
    #     if trade.is_open and trade.entry_side == 'buy':
    #         if (cn1['close'] < cn1['ema100']) and (cn2['close'] > cn2['ema100']):
    #         # if cn1['sroc'] > 0 and cn2['sroc'] < 0:
    #             return 'exit_long_cross'
        
    #     if trade.is_open and trade.entry_side =='sell':
    #         if (cn1['close'] > cn1['ema100']) and (cn2['close'] < cn2['ema100']):
    #         # if cn1['sroc'] < 0 and cn2['sroc'] > 0:
    #             return 'exit_short_cross'
        
    #     return None

    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str,side: str,**kwargs) -> float:
        
        return 5.0

    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,proposed_stake: float, min_stake: float | None, max_stake: float,leverage: float, entry_tag: str | None, side: str,**kwargs) -> float:

    #     # balance = self.wallets.get_all_balances()
    #     stake = proposed_stake * 0.5
    #     if stake < min_stake:
    #         return proposed_stake
        
    #     if stake > max_stake:
    #         return proposed_stake
        
    #     if min_stake<stake < max_stake:
    #         return stake