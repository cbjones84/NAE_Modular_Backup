"""
Auto-implemented improvement from GitHub
Source: cryptometa-source/FromZeroToBotYoutube/BotTests.py
Implemented: 2025-12-09T11:16:05.246865
Usefulness Score: 100
Keywords: def , class , strategy, simulate, fit, loss, stop, loss
"""

# Original source: cryptometa-source/FromZeroToBotYoutube
# Path: BotTests.py


# Function: test_PnlTradingEngine
def test_PnlTradingEngine():    
    test_setup = TestSetup()

    engine = PnlTradingEngine(test_setup.token_info, test_setup.order_executor, test_setup.order)

    engine.start()

    #Check if limit order or stop order triggers
    engine._process_event_task()

    assert engine.state == StrategyState.PENDING

    limit_price = (1+test_setup.profit_limit.trigger_at_percent.ToUiValue()/100)*test_setup.base_token_price.ToUiValue()
    stop_price = (1+test_setup.stop_loss.trigger_at_percent.ToUiValue()/100)*test_setup.base_token_price.ToUiValue()
    
    #Force Limit Price
    test_setup.market_manager.current_price = limit_price
    engine._process_event_task()
    
    assert engine.state == StrategyState.COMPLETE

    #Reset State
    engine.state = StrategyState.PENDING

    #Force Stop Loss Price
    test_setup.market_manager.current_price = stop_price
    engine._process_event_task()
    
    assert engine.state == StrategyState.COMPLETE
    

