import ccxt
from pprint import pp
hp = ccxt.hyperliquid()
bi = ccxt.binance()
by = ccxt.bybit()
hp.options['defaultType'] ='swap'
hp.apiKey= "0x5b0129E2381E4A4Fe2c78acAD160065ff349669B"
# hp.secret = "0xc391d6de84a84d85af618e2b8a670ab18e61008c99e867d97985a0b80bd8cd81"
hp.walletAddress = "0xE797807894500fA49f0614DD88aDfa6bB8320b9E"

pair = 'KAS/USDC:USDC'

markets = bi.load_markets()
market = ""
for market in markets:
    if market.endswith('USDC:USDC'):
        print(market)

# markets = hp.load_markets()
# for market in markets:
#     if market == 'ARB/USDC:USDC':
#         print(market)
# print(markets)
# print(len(markets.keys()))

# tickers = hp.fetch_tickers()
# for ticker in tickers:
#     print(ticker)