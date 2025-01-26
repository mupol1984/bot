import pandas_ta as ta
import pandas as pd
import numpy as np




def bb(source:pd.Series , len:int=20,std:float=2.0)-> pd.Series:
    """
    lower , meddium , upper
    """
    b = ta.bbands(close= source , length= len , std= std)
    lower:pd.Series = b[b.columns[0]]
    meddium:pd.Series = b[b.columns[1]]
    upper:pd.Series = b[b.columns[2]]
    return lower , meddium , upper

def qqe(source:pd.Series , len:int=14, sm:int=5 , fc:float= 4.32)->pd.Series:
    """
    QQE , QQEL , QQES
    """
    q = ta.qqe(close= source , length= len , smooth= sm , factor= fc)
    qqe:pd.Series = q[q.columns[0]]
    qqel:pd.Series = q[q.columns[2]]
    qqes:pd.Series = q[q.columns[3]]
    return qqe , qqel , qqes




def mfi(dataframe:pd.DataFrame , len:int=14)-> pd.Series:
    """
    mfi
    """
    mfi = ta.mfi(high= dataframe['high'] , low= dataframe['low'] , close= dataframe['close'] , volume= dataframe['volume'] , length= len)
    return mfi




def cmf(dataframe:pd.DataFrame , len:int = 14 )-> pd.Series:
    """
    cmf
    """
    cmf = ta.cmf(high=dataframe['high'] , low= dataframe['low'] , close= dataframe['close'] , volume=dataframe['volume'] , length= len)
    return cmf


def atr(dataframe:pd.DataFrame , len:int = 14)-> pd.Series:
    """
    atr
    """
    atr = ta.atr(high=dataframe['high'] , low= dataframe['low'] , close= dataframe['close'] , length= len)
    return atr



def donchain(dataframe:pd.DataFrame , lower_len:int = 5,upper_len:int = 5)-> pd.Series:
    """
    lower_band , middle_band , upper_band
    """
    don = ta.donchian(high= dataframe['high'] , low=dataframe['low'] , lower_length= lower_len , upper_length= upper_len)
    l = don[don.columns[0]]
    m = don[don.columns[1]]
    u = don[don.columns[2]]
    return l , m , u



def true_range(dataframe:pd.DataFrame)-> pd.Series:
    """
    treue_range
    """
    tr = ta.true_range(high=dataframe['high'] , low=dataframe['low'] , close=dataframe['close'])
    return tr



def cgar(dataframe:pd.DataFrame)->float:
    """
    cgar:float
    """
    cg = ta.cagr(close= dataframe['close'])
    return cg


def calmar_ratio(dataframe:pd.DataFrame , len:int=3)-> float:
    """
    calmar ration :float
    """
    cr = ta.calmar_ratio(close=dataframe['close'] ,years= len)
    return cr



def adx(dataframe:pd.DataFrame , len:int = 14 , sig:int = 14)->pd.Series:
    """
    adx , dm+ , dm-
    """
    a = ta.adx(high=dataframe['high'] , low= dataframe['low'] , close= dataframe['close'] , length= len , lensig= sig)
    adx = a[a.columns[0]]
    d_plus = a[a.columns[1]]
    d_minus = a[a.columns[2]]
    return adx , d_plus , d_minus



def supertrend(dataframe:pd.DataFrame , period:int = 5 , multiplyer:float = 3):
    t = ta.supertrend(high=dataframe['high'] , low=dataframe['low'] , close=dataframe['close'] , length= period , multiplier= multiplyer)
    trend = t[t.columns[0]]
    direction = t[t.columns[1]]
    long = t[t.columns[2]]
    short = t[t.columns[3]]

    return trend , direction , long , short

