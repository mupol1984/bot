from freqtrade.configuration import Configuration
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from pathlib import Path
from freqtrade.resolvers import StrategyResolver
from freqtrade.data.dataprovider import DataProvider
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
# import plotly.io as pio
# pio.renderers.default = "browser"

pd.set_option('display.max_columns', None)

pair = "ARB/USDT:USDT"


config_path = Path("c:\\bot\\user_data\\download.json")
data_dir = "c:\\bot\\user_data"
config = Configuration.from_files([config_path])
config['timeframe'] = '1m'
config["strategy"] = "hassan"
data_location = config["datadir"]

candel = load_pair_history(datadir=data_location,timeframe=config["timeframe"],pair=pair,data_format = "json",candle_type=CandleType.FUTURES)

strategy = StrategyResolver.load_strategy(config)
strategy.dp = DataProvider(config, None, None)
strategy.ft_bot_start()

df = strategy.analyze_ticker(candel, {'pair': f"{pair}"})
df = df.set_index('date', drop=False)

df = df.tail(300)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True , shared_yaxes= True, vertical_spacing=0.001 ,horizontal_spacing= 0.001 , row_heights=[0.6, 0.2,0.2])
fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'] , name='Candlestick'), row=1, col=1)

fig.update_layout(template='plotly_dark',
                  title='Candlestick Chart',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False,
                  plot_bgcolor='black',
                  paper_bgcolor='black')

config = {'scrollZoom': True , 'editable0':True , 'autosizable':True , 'responsive':True , 'fillFrame':True}
fig.show(config=config)

