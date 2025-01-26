
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

def chaikin_money_flow(dataframe:DataFrame, length=20, fillna=False) -> Series:
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(length, min_periods=0).sum() / df['volume'].rolling(length, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


class kol(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    max_open_trades = 3
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

    use_custom_stoploss = True
    bbdelta_close = DecimalParameter(0.0005, 0.02, default=0.02, decimals=4, space='buy', optimize=True)
    bbdelta_tail = DecimalParameter(0.7, 1.0, default=0.7, decimals=4, space='buy', optimize=True)
    close_bblower = DecimalParameter(0.0005, 0.02, default=0.02, decimals=4, space='buy', optimize=True)
    closedelta_close = DecimalParameter(0.0005, 0.02, default=0.02, decimals=4, space='buy', optimize=True)
    rocr_1h = DecimalParameter(0.5, 1.0, default=0.5, decimals=4, space='buy', optimize=True)

    leverage_num = IntParameter(low=1, high=10, default=1, space='buy', optimize=True)

    sell_fisher = DecimalParameter(0.1, 0.5, default=0.5, decimals=4, space='sell')
    sell_bbmiddle_close = DecimalParameter(0.97, 1.1, default=1.1, decimals=4, space='sell')

    pHSL = DecimalParameter(-0.990, -0.040, default=-0.08, decimals=3, space='sell')
    pPF_1 = DecimalParameter(0.008, 0.100, default=0.016, decimals=3, space='sell')
    pSL_1 = DecimalParameter(0.008, 0.100, default=0.011, decimals=3, space='sell')
    pPF_2 = DecimalParameter(0.040, 0.200, default=0.080, decimals=3, space='sell')
    pSL_2 = DecimalParameter(0.040, 0.200, default=0.040, decimals=3, space='sell')

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,current_rate: float, current_profit: float, **kwargs) -> float:
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if self.can_short:
            if (-1 + ((1 - sl_profit) / (1 - current_profit))) <= 0:
                return 1
        else:
            if (1 - ((1 + sl_profit) / (1 + current_profit))) <= 0:
                return 1

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid
        
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        
        return dataframe
    

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[(dataframe['rocr_1h'].gt(self.rocr_1h.value)) & (
                    ((dataframe['lower'].shift().gt(0)) &
                    (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.bbdelta_close.value)) &
                    (dataframe['closedelta'].gt(dataframe['ha_close'] * self.closedelta_close.value)) &
                    (dataframe['tail'].lt(dataframe['bbdelta'] * self.bbdelta_tail.value)) &
                    (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                    (dataframe['ha_close'].le(dataframe['ha_close'].shift()))) | 
                    ((dataframe['ha_close'] < dataframe['ema_slow']) &
                    (dataframe['ha_close'] < self.close_bblower.value * dataframe['bb_lowerband']))),'enter_long'] = 1

        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['fisher'] > self.sell_fisher.value) &
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_fast'] > dataframe['ha_close']) &
            ((dataframe['ha_close'] * self.sell_bbmiddle_close.value) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0),'exit_long'] = 1
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value

    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,proposed_stake: float, min_stake: float | None, max_stake: float,leverage: float, entry_tag: str | None, side: str,**kwargs) -> float:
    #     stake = self.wallets.get_total_stake_amount() / self.max_open_trades
    #     if stake < min_stake:
    #         return min_stake
        
    #     if stake > max_stake:
    #         return max_stake
        
    #     if min_stake<stake < max_stake:
    #         return stake
