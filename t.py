import ccxt
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import indicators as ind
bi = ccxt.binance()

pair = "ARB/USDT:USDT"

data = bi.fetch_ohlcv(pair, timeframe='5m', limit=500)

df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

df['date'] = pd.to_datetime(df['date'], unit='ms')

df['pcup'] , df['pcdown'] = ind.PCC(df , 20 ,2)

df['sldown'], df['slup'] = ind.SSLChannels_ATR(df , 10)

df['v_lower'], df['vwap'], df['v_upper'] = ind.VWAPB(df)



fig = make_subplots(rows=3, cols=1, shared_xaxes=False , row_heights=[0.5, 0.25,0.25])

#@ ROW 1
fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'] , name='Candlestick'), row=1, col=1)
# fig.add_trace(go.Scatter(x=df['date'], y=df['zema'], mode='lines', name='zema' , line=dict(width=1 , color='#BDBDBD')) , row= 1 , col=1)
# fig.add_trace(go.Scatter(x=df['date'], y=df['zema'], mode='lines', name='zema' , line=dict(width=1 , color='#BDBDBD')) , row= 1 , col=1)
# fig.add_trace(go.Scatter(x=df['date'], y=df['zema'], mode='lines', name='zema' , line=dict(width=1 , color='#BDBDBD')) , row= 1 , col=1)


#@ ROW 2
fig.add_trace(go.Scatter(x=df['date'], y=df['pcup'], mode='lines', name='pcup' , line=dict(width=1 , color='#F57F17')) , row= 2 , col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['pcdown'], mode='lines', name='pcdown' , line=dict(width=1 , color='#F57F17')) , row= 2 , col=1)
# fig.add_trace(go.Scatter(x=df['date'], y=df['fastsma'], mode='lines', name='fastsma' , line=dict(width=1 , color='#F57F17')) , row= 2 , col=1)

#@ ROW 3
fig.add_trace(go.Scatter(x=df['date'], y=df['sldown'], mode='lines', name='sldown' , line=dict(width=1 , color='#F57F17')) , row= 3 , col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['slup'], mode='lines', name='slup' , line=dict(width=1 , color='#F57F17')) , row= 3 , col=1)
# fig.add_trace(go.Scatter(x=df['date'], y=df['hull_mom'], mode='lines', name='hull_mom' , line=dict(width=1 , color='#F44336')) , row= 3 , col=1)


# long_entries = df[df['enter_long'] == 1]
# fig.add_trace(go.Scatter(x=long_entries['date'], y=long_entries['close'], mode='markers', name='Enter Long', marker=dict(color='green', size=20, symbol='triangle-up')))

# short_entries = df[df['exit_long'] == 1]
# fig.add_trace(go.Scatter(x=short_entries['date'], y=short_entries['close'],mode='markers', name='Enter Short', marker=dict(color='red', size=20, symbol='triangle-down')))



fig.update_layout(template='plotly_dark',
                  xaxis_title_font_size=6,
                  yaxis_title_font_size=6,
                #   title='Candlestick Chart',
                  xaxis_title='',
                  yaxis_title='',
                  xaxis_rangeslider_visible=False,
                  plot_bgcolor='black',
                  paper_bgcolor='black')

config = {'scrollZoom': True , 'editable':True , 'autosizable':True , 'responsive':True , 'fillFrame':True}
fig.show(config=config)