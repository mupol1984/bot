from freqtrade.configuration import Configuration
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.resolvers import StrategyResolver
from freqtrade.data.dataprovider import DataProvider
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from pathlib import Path
from pandas import DataFrame , Series
import numpy as np
from datetime import datetime , timedelta

pd.set_option('display.max_columns', None)


timeframe = '5m'
pair = "ARB/USDT:USDT"
candle_number = 800
strategy_name = 'ind'
height = 1800
width = 1600

long = False
short = False


#!   ROW  1
ind1 = ''
color1 = '#ffffff'

ind2 = ''
color2 = '#ffff00'   # yellow

ind3 = ''
color3 = '#ff8000'    # orange

ind4 = ''
color4 = '#ff0080'

ind5 = ''
color5 = '#ff8080'

ind6 = ''
color6 = '#ff0000'

#!   ROW  2
ind7 = 'roc'
color7 = '#00ff00'

ind8 = ''
color8 = '#ff0000'


#!   ROW  3
ind9 = 'sroc'
color9 = '#00ff00'

ind10 = ''
color10 = '#ff0000'


#!   ROW  4
ind11 = ''
color11 = ''

ind12 = ''
color12 = ''

ind13 = ''
color13 = ''

ind14 = ''
color14 = ''







config_path = Path("c:\\freq\\user_data\\download.json")
config = Configuration.from_files([config_path])
config['timeframe'] = timeframe
config["strategy"] = strategy_name
data_location = config["datadir"]
candel = load_pair_history(datadir=data_location,timeframe=config["timeframe"],pair=pair,data_format = "json",candle_type=CandleType.FUTURES)
strategy = StrategyResolver.load_strategy(config)
strategy.dp = DataProvider(config, None, None)
strategy.ft_bot_start()
df = strategy.analyze_ticker(candel, {'pair': pair})
df['date'] = pd.to_datetime(df['date'])
# df = df.set_index('date', drop=False)
df = df.tail(candle_number)


fig = make_subplots(rows=5, cols=1, row_heights=[0.3, 0.18 , 0.18 , 0.18 , 0.18] , vertical_spacing=0.05 , horizontal_spacing= 0.05)


#!    ROW  1
fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'] , name='Candlestick'), row=1, col=1)

if ind1!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind1], mode='lines', name=ind1 , line=dict(width=1.5 , color=color1)) , row= 1 , col=1)

if ind2!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind2], mode='lines', name=ind2 , line=dict(width=1.5 , color=color2)) , row= 1 , col=1)

if ind3!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind3], mode='lines', name=ind3 , line=dict(width=1.5 , color=color3)) , row= 1 , col=1)

if ind4!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind4], mode='lines', name=ind4 , line=dict(width=1 , color=color4)) , row= 1 , col=1)

if ind5!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind5], mode='lines', name=ind5 , line=dict(width=1 , color=color5)) , row= 1 , col=1)

if ind6!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind6], mode='lines', name=ind6 , line=dict(width=1 , color=color6)) , row= 1 , col=1)


#!    ROW  2

if ind7!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind7], mode='lines', name=ind7 , line=dict(width=1 , color=color7)) , row= 2 , col=1)

if ind8!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind8], mode='lines', name=ind8 , line=dict(width=1 , color=color8)) , row= 2 , col=1)
  

#!    ROW 3

if ind9!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind9], mode='lines', name=ind9 , line=dict(width=1 , color=color9)) , row= 3 , col=1)

if ind10!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind10], mode='lines', name=ind10 , line=dict(width=1 , color=color10)) , row= 3 , col=1)


#!    ROW 4

if ind11!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind11], mode='lines', name=ind11 , line=dict(width=1 , color=color11)) , row= 4 , col=1)

if ind12!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind12], mode='lines', name=ind12 , line=dict(width=1 , color=color12)) , row= 4 , col=1)


#!    ROW 5

if ind13!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind13], mode='lines', name=ind13 , line=dict(width=1 , color=color13)) , row= 5 , col=1)

if ind14!= '':
    fig.add_trace(go.Scatter(x=df['date'], y=df[ind14], mode='lines', name=ind14 , line=dict(width=1 , color=color14)) , row= 5 , col=1)



if long:
    long_entries = df[df['enter_long'] == 1]
    fig.add_trace(go.Scatter(x=long_entries['date'], y=long_entries['close'], mode='markers', name='Enter Long', marker=dict(color='green', size=20, symbol='triangle-up')))

if short:
    short_entries = df[df['enter_short'] == 1]
    fig.add_trace(go.Scatter(x=short_entries['date'], y=short_entries['close'],mode='markers', name='Enter Short', marker=dict(color='red', size=20, symbol='triangle-down')))

fig.update_layout(height=height, width=width , showlegend=True)

fig.update_layout(template='plotly_dark',
                  plot_bgcolor='black',
                  paper_bgcolor='black',
                  xaxis_rangeslider_visible=False)

config = {'scrollZoom': True , 'editable':True , 'autosizable':True , 'responsive':True , 'fillFrame':True}
fig.show(config=config)


