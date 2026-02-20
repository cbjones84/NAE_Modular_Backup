"""
Auto-implemented improvement from GitHub
Source: njagwani/Python-Crypto-Trading-Bot/bot.py
Implemented: 2025-12-09T11:04:36.482985
Usefulness Score: 80
Keywords: def , calculate, position
"""

# Original source: njagwani/Python-Crypto-Trading-Bot
# Path: bot.py


# Function: on_message
def on_message(ws, message):
    global closes 

    print('received message')
    print(message)
    json_message = json.loads(message)
    pprint.pprint(json_message)
    
    candle = json_message['k']

    is_candle_closed = candle['x']
    close = candle['c']

    if is_candle_closed:
        print("candle closed at {}".format(close))
        closes.append(float(close))
        print("closes")
        print(closes)

        if len(closes) > RSI_PERIOD: 
            np_closes  = numpy.array(closes)
            rsi = talib.RSI(np_closes, RSI_PERIOD)
            print("all rsis calculated so far")
            print(rsi)
            last_rsi = rsi[-1]
            print("the current rsi is {}".format(last_rsi))

            if last_rsi > RSI_OVERBOUGHT:
                if in_position:
                    print("Overbought! Sell! Sell! Sell!")
                    #put binance sell logic here
                    order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = False
                else:
                    print("It is overbought, but we dont own any, Nothing to do")
            
            if last_rsi < RSI_OVERSOLD:
                if in_position:
                    print("It is oversold, but you already own it, nothing to do.")
                else:
                    print("Oversold! Buy! Buy! Buy!")
                    #put binance buy order logic here
                    order_succeeded = order (SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = True


