
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame,Series
from typing import Dict, Optional, Union, Tuple , Any

from freqtrade.strategy import (IStrategy,informative,
    BooleanParameter,CategoricalParameter,DecimalParameter,IntParameter,RealParameter,timeframe_to_minutes,timeframe_to_next_date,timeframe_to_prev_date,merge_informative_pair,stoploss_from_absolute,stoploss_from_open)

import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from technical.qtpylib import crossed_above , crossed_below
from freqtrade.persistence import Trade , Order
from technical.qtpylib import crossed_above , crossed_below
from functools import reduce

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

def WILLIAMS(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series((highest_high - dataframe["close"]) / (highest_high - lowest_low), name="{0} Williams %R".format(period))
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



class javad(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    max_open_trades = 3
    can_short:bool = True

    #################################################
    # Buy hyperspace params:
    buy_params = {
        "buy_cmf": -0.35,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_cmf": 0.04,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.21,
        "36": 0.061,
        "92": 0.028,
        "199": 0
    }

    # Stoploss:
    stoploss = -0.249
    ################################################

    # minimal_roi = {
    #     "0": 0.9
    # }

    # stoploss = -0.9

    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 500

    buy_cmf = DecimalParameter(-0.5 , -0.01 , default= -0.12 , decimals= 2 , space='buy')
    sell_cmf = DecimalParameter(0.01 , 0.5 , default= 0.12 , decimals= 2 , space='sell')

    # @informative('5m')
    # def populate_indicators_5m(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:

    #     df['ema25'] = pta.ema(df['close'] , 25)
    #     df['ema50'] = pta.ema(df['close'] , 50)
    #     df['ema100'] = pta.ema(df['close'] , 100)
    #     df['ema150'] = pta.ema(df['close'] , 150)
    #     df['ema200'] = pta.ema(df['close'] , 200)
    #     df['cmf'] = chaikin_money_flow(df , 24)
    #     return df




    def populate_indicators(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        df['tema'] = pta.tema(df['close'] , 6)
        df['ema25'] = pta.ema(df['close'] , 25)
        df['ema50'] = pta.ema(df['close'] , 50)
        df['ema75'] = pta.ema(df['close'] , 75)
        df['ema100'] = pta.ema(df['close'] , 100)
        df['ema125'] = pta.ema(df['close'] , 125)
        df['ema150'] = pta.ema(df['close'] , 150)
        df['ema200'] = pta.ema(df['close'] , 200)
        df['cmf'] = chaikin_money_flow(df , 24)
        return df


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        
        df.loc[(
                (df['close'] > df['ema75']) &
                (df['close'] < df['ema125']) &
                (df['ema75'] < df['ema125']) &
                # (df['ema100'] < df['ema200']) &
                (df['cmf'] < self.buy_cmf.value) &
                (df['volume'] > 0)) , 'enter_long'] = 1

        df.loc[(
                (df['close'] < df['ema75']) &
                (df['close'] > df['ema125']) &
                (df['ema75'] > df['ema125']) &
                # (df['ema100'] > df['ema200']) &
                (df['cmf'] > self.sell_cmf.value) &
                (df['volume'] > 0)) , 'enter_short'] = 1



        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (qtpylib.crossed_below(df['close'] , df['ema150']) &
             (df['volume'] > 0)) , 'exit_long'] = 1

        df.loc[
            (qtpylib.crossed_above(df['close'] , df['ema150']) &
             (df['volume'] > 0)) , 'exit_short'] = 1
     
        # df.loc[(df['volume'] > 0) , 'exit_long'] = 0
        # df.loc[(df['volume'] > 0) , 'exit_short'] = 0
        return df


    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        cn1 = dataframe.iloc[-1].squeeze()
        cn2 = dataframe.iloc[-2].squeeze()
        if trade.is_open and trade.entry_side == 'buy':
            if (cn1['close'] < cn1['ema150']) and (cn2['close'] > cn2['ema150']):
            # if cn1['sroc'] > 0 and cn2['sroc'] < 0:
                return 'exit_long_cross'
        
        if trade.is_open and trade.entry_side =='sell':
            if (cn1['close'] > cn1['ema150']) and (cn2['close'] < cn2['ema150']):
            # if cn1['sroc'] < 0 and cn2['sroc'] > 0:
                return 'exit_short_cross'
        
        return None



    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str,side: str,**kwargs) -> float:
        
        return 5.0

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,proposed_stake: float, min_stake: float | None, max_stake: float,leverage: float, entry_tag: str | None, side: str,**kwargs) -> float:

        # balance = self.wallets.get_all_balances()
        stake = self.wallets.get_total_stake_amount() / self.max_open_trades
        if stake < min_stake:
            return min_stake
        
        if stake > max_stake:
            return max_stake
        
        if min_stake<stake < max_stake:
            return stake