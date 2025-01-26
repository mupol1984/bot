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
backtest_path = os.path.join(user_data_path,'backtest_results')
hyper_path = os.path.join(user_data_path , 'data\\hyperliquid\\futures')
binance_path = os.path.join(user_data_path , 'data\\binance\\futures')

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

def hyperopt(strategy_name:str = None , hyperopt_loss:str = 'OnlyProfitHyperOptLoss' , epoch:int = 150 , day_timerange:int = 20):
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
    return ['hyperopt', '-c',download_config_path,'--userdir',user_data_path , '-s' , strategy_name ,'--spaces' ,'buy' , 'sell' , 'roi' , 'stoploss' , '--hyperopt-loss' , hyperopt_loss , '-e' , str(epoch) , '--timerange', time_range , '--ignore-missing-spaces' ,'--disable-param-export' ]



#! strayegy update
# args = strategy_update()
# main.main(args)

#! backtesting show
# args = backtesting_show()
# main.main(args)

#! hyper opt list
# args = hyperopt_list('hosna1')
# main.main(args)

#! install ui
# args = install_ui()
# main.main(args)

#! download  data
# args = download_data(exchange='bybit' , days=35 , timeframe='5m')
# main.main(args)

# ! hyper opt
args = hyperopt(strategy_name='hosna', hyperopt_loss='OnlyProfitHyperOptLoss' , day_timerange=20)
main.main(args)

#! hyper opt
# args = hyperopt(strategy_name='met' ,hyperopt_loss='ShortTradeDurHyperOptLoss' , day_timerange= 20 )
# main.main(args)

# ! backtest
# args = backtest(strategy='hosna' , day=30 )
# main.main(args)

# move_folder_contents(binance_path , hyper_path)


#! plotting
# args = plot_dataframe(strategy=mystrategy , pair=pair)
# main.main(args)

#! new strategy
# args = new_strategy(strategy="ind" , template='advanced')
# main.main(args)

#! test pairlist
# args = test_pairlist()
# main.main(args)

#! trade
# args = trade(strategy='farshad')
# main.main(args)