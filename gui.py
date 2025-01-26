import FreeSimpleGUI as sg
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
from freqtrade import main
import shutil

file_path = os.path.abspath(__file__)
folder_path = os.path.dirname(file_path)
user_data_path = os.path.join(folder_path, 'user_data')
# user_data_path = Path("c:\\hossain\\user_data")
config_file = os.path.join(user_data_path, 'config.json')
pair_config_file = os.path.join(user_data_path, 'pair.json')
download_config_path = os.path.join(user_data_path, 'download.json')
data_path = os.path.join(user_data_path,'data')

config:dict = {}

def move_folder_contents(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        src_file = os.path.join(source_folder, filename)
        dst_file = os.path.join(destination_folder, filename)
        shutil.move(src_file, dst_file)
        print(f"Moved {src_file} to {dst_file}")
    return
   

def timerange(day:int= 30):
    end_date = datetime.now()- timedelta(days=1)
    start_date = end_date - timedelta(days= day)
    return f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

def read_config(config_path:Path):
    with open(config_path , 'r') as file:
        conf:dict = json.load(file)
    return conf

def write_config(conf:dict , config_path:Path):
    with open(config_path , 'w') as file:
        json.dump(conf, file , indent=4)
    return

def del_folder_content(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            shutil.rmtree(dir_path)
    return


def edge(strategy:str , days:int = 30):
    time_range = timerange(day= days)
    return ['edge' ,'-c'  ,config_file,'--userdir',user_data_path , '-s' , strategy , '--timerange' , time_range ]


def install_ui():
    # args = ['install-ui' , '--erase']
    # args = ['install-ui' , '--ui-version' , '1.3.2']
    # args = ['install-ui' , '--help']
    args = ['install-ui']
    return args

def backtesting_show():
    return ['backtesting-show' , '-c',download_config_path, '--userdir',user_data_path]

def plot_profit(strategy:str ,day_timerange:int = 10 ):
    time_range = timerange(day= day_timerange)
    return ['plot-profit' , '-c',download_config_path,'--userdir',user_data_path , '--timerange', time_range , '-s' , strategy ]

def hyperopt_list(min_trades:int = 1):
    return ['hyperopt-list' , '-c',download_config_path,'--userdir',user_data_path , '--best' , '--profitable' , '--min-trades' , str(min_trades)]

def strategy_update():
    return ['strategy-updater' , '-c',config_file,'--userdir',user_data_path]

def webserver():
    return ['webserver' , '-c',config_file,'--userdir',user_data_path]

def trade(strategy:str):
    args = ['trade' , '-c',config_file,'--userdir',user_data_path , '-s' , strategy ]
    return args

def new_strategy(strategy:str = None , template:str = 'full'):
    """
        full
        minimal
        advanced
    """
    return ['new-strategy' ,'--userdir',user_data_path , '--template' , template , '-s' , strategy]

def test_pairlist():
    return ['test-pairlist' , '-c', pair_config_file , '--userdir',user_data_path  , '--one-column']

def plot_dataframe(strategy:str= None , pair:str = None , days:int = 5):
    time_range = timerange(day=days)
    return ['plot-dataframe' , '-c',download_config_path,'--userdir',user_data_path ,'-s' , strategy , '-p' , pair , '--timerange', time_range]

def download_data(exchange:str='binance' , days:int = 35, timeframe:str = '5m' ):
    time_range = timerange(day=days)
    return ['download-data' , '-c',download_config_path,'--userdir',user_data_path , '--exchange' , exchange , '--timerange', time_range , '-t' , timeframe]


def backtest(strategy:str = None , day:int = 10 ):
    time_range = timerange(day=day)
    return ['backtesting' , '-c',download_config_path,'--userdir',user_data_path , '--timerange', time_range ,'-s' , strategy ]

def hyperopt(strategy_name:str = None , hyperopt_loss:str = 'OnlyProfitHyperOptLoss' , epoch:int = 100 , day_timerange:int = 30):
    """
        ShortTradeDurHyperOptLoss
        OnlyProfitHyperOptLoss
        SharpeHyperOptLoss
        CalmarHyperOptLoss
        # MaxDrawDownHyperOptLoss
        MaxDrawDownRelativeHyperOptLoss
        SharpeHyperOptLossDaily
        SortinoHyperOptLoss
        SortinoHyperOptLossDaily
         '--ignore-missing-spaces'
         ,'--analyze-per-epoch'
          ,'--disable-param-export'
    """
    time_range = timerange(day=day_timerange)
    return ['hyperopt', '-c',download_config_path,'--userdir',user_data_path , '-s' , strategy_name ,'--spaces' ,'all' , '--hyperopt-loss' , hyperopt_loss , '-e' , str(epoch) , '--timerange', time_range ,'--ignore-missing-spaces']

theme = sg.theme('Black2')

hyperopts = ['ShortTradeDurHyperOptLoss' , 'OnlyProfitHyperOptLoss' , 'CalmarHyperOptLoss' , 'SharpeHyperOptLoss' , 'MaxDrawDownRelativeHyperOptLoss' , 'SortinoHyperOptLoss']
templates = ['full' , 'advanced' , 'minimal']
timeframes = ['1m' , '5m' , '15m' , '30m' , '1h' , '2h' , '4h']

layout = [
    [sg.Push(),sg.Text("Select an action:") , sg.Push()],
    [sg.Button("Test Pairlist" , size= 12)],

    #!  install ui
    [sg.Button("Install UI" , size=12),sg.Checkbox('--erase' , default= False , size= 10) ],

    #! New strategy
    [sg.Button("New Strategy" , size= 12) , sg.Text('Strategy:' , size= 10) , sg.Input('' , size= 12) , sg.Text('Template:' , size= 10 )  , sg.Combo(values= templates , default_value= templates[2] , size= 10 )],

    #!  Download Data
    [sg.Button("Download Data" , size= 12) , sg.Text('Exchange:' , size= 10) ,sg.Input('bybit' , size= 12) ,sg.Combo(values= timeframes , default_value= timeframes[1] , size= 10 ) , sg.Text('Days:' , size= 8) , sg.Input('30' , size= 10)],

    #!  Backtest
    [ sg.Button("Backtest" , size= 12) , sg.Text('Strategy:' , size= 10) , sg.Input('' , size= 12) , sg.Text('Days:' , size= 8) , sg.Input('30' , size= 10) ],

    #!  Hyperopt
    [sg.Button("Hyperopt" , size= 12) , sg.Text('Strategy:' , size= 10) , sg.Input('' , size= 12) , sg.Text('Days:' , size= 8) , sg.Input('20' , size= 10)],
    
]

# Create the window
window = sg.Window("FREQTRADE", layout= layout , size=(600,400))

# Event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == "Install UI":
        args = install_ui()

    elif event == "Hyperopt":
        args = hyperopt(strategy_name="example_strategy")

    elif event == "Backtest":
        args = backtest(strategy="example_strategy")

    elif event == "New Strategy":
        args = trade(strategy="example_strategy")

    elif event == "Download Data":
        args = download_data()

window.close()