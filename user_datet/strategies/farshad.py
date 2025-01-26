# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


def CMF(dataframe:DataFrame, length=20, fillna=False) -> Series:
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(length, min_periods=0).sum() / df['volume'].rolling(length, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

class farshad(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_short: bool = True
    max_open_trades = 2
    #######################



    ############################
    minimal_roi = {
        "0":0.9
    }
    stoploss = -0.25
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 400

    # buy_rsi = IntParameter(10, 40, default=30, space="buy")
    # sell_rsi = IntParameter(60, 90, default=70, space="sell")

    buy_cmf = DecimalParameter(-0.4, -0.01, decimals=2 ,default=-0.15, space="buy")
    sell_cmf = DecimalParameter(0.1,0.4, decimals=2 ,default=0.15, space="sell")

   
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'ema5': {'color': 'white'},
                'ema50': {'color': 'blue'},
                'ema100':{'color': 'green'},
                'ema150': {'color': 'purple'},
                'ema200': {'color': 'orange'},
            },
            'subplots': {
                "CMF": {
                    'cmf': {'color': 'orange'},
                }
            }
        }


    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:

        df['tema'] = pta.tema(df['close'] , 6)
        df['ema25'] = pta.ema(df['close'] , 25)
        df['ema50'] = pta.ema(df['close'] , 50)
        df['ema125'] = pta.ema(df['close'] , 125)
        df['ema100'] = pta.ema(df['close'] , 100)
        df['ema150'] = pta.ema(df['close'] , 150)
        df['ema200'] = pta.ema(df['close'] , 200)
        df['cmf'] = CMF(df , 20)
        # df['rsi'] = pta.rsi(df['close'] , 14)

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['tema'] < df['ema50']) &  # Signal: RSI crosses above buy_rsi
                (df['ema50'] > df['ema100']) &  # Guard: tema below BB middle
                (df['ema100'] > df['ema200']) &
                (df['cmf'] < self.buy_cmf.value) &  # Guard: tema below BB middle
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1
        
        df.loc[
            (
                (df['tema'] > df['ema50']) &  # Signal: RSI crosses above buy_rsi
                (df['ema50'] < df['ema100']) &  # Guard: tema below BB middle
                (df['ema100'] < df['ema200']) &
                (df['cmf'] > self.sell_cmf.value) &  # Guard: tema below BB middle
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1

        return df


    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (qtpylib.crossed_above(df['tema'] , df['ema100'])) &
                # (df['cmf'] >= -0.2) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1
        
        df.loc[
            (
                (qtpylib.crossed_below(df['tema'] , df['ema100'])) &
                # (df['cmf'] <= 0.2 ) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1


        return df


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,proposed_stake: float, min_stake: float | None, max_stake: float,leverage: float, entry_tag: str | None, side: str,**kwargs) -> float:

        # balance = self.wallets.get_all_balances()
        stake = proposed_stake * 0.6

        if stake >= min_stake and stake <= max_stake:
            return stake

        if stake < min_stake:
            return proposed_stake
        
        if stake > max_stake:
            return proposed_stake
        
        

    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str,side: str,**kwargs) -> float:
        
        return 5.0