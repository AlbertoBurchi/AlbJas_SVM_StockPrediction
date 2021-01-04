# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:26:34 2020

@author: alber
"""
import get_tickers as get_tickers
import back_test as back_test
import pandas as pd
import sample_slopes as sample_slopes
import numpy as np
import settings
from performance_array import performance_stream_callable


def test_performance_array(ticker,batch,look_ahead):
    """
    This test is used to try out the returns calculator on the stock market data for a single stock (change ticker)
    """
#    ticker='MSFT'
    data = {
        'col4CLS': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.1, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4slope_sum': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.2, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4bid_stream': [None, None, None, None, None,1 , 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    stock_data = pd.DataFrame(data=data)

    print(len(data['col4CLS']))
    print(len(data['col4bid_stream']))
    
    Back_Test = back_test.BackTest(stock_data, settings.settings_dict['test_model_path'])
        
    algorithm_return = Back_Test.take_bid_stream_calculate_profit("col4bid_stream", 3, 2, for_graph=True)
    
    array_of_bid_stream = Back_Test.main_df["col4bid_stream"].tolist()[3 + 2 - 1:] #1 or 0 long or short
    index_bid_stream = list(range(0, len(array_of_bid_stream)))

    runningTotal = []
    total = 0
    for n in algorithm_return:
        total += n
        runningTotal.append(total)
    
    # list_of_bids = Back_Test.array_of_profits
    # shift the graph to the left to account for the initial days there there
    # inst enough info for
    index = list(range(0, len(runningTotal)))
    
    index_stock = list(range(0, len(stock_data['col4CLS'].tolist()[3 + 2 - 1:])))

    stock_price=stock_data['col4CLS'].tolist()[3 + 2 - 1:]
    
    stock_log_return_stream=Back_Test.test_calculate_holding_log_return('col4CLS')
    stock_log_return=Back_Test.calculate_holding_logreturn('col4CLS',3,2)
    stock_log_return_stream=stock_log_return_stream[3 + 2 - 1:]
    algo_log_return_stream= Back_Test.take_bid_stream_calculate_log_return("col4bid_stream", 3, 2)
  
