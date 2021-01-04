# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:49:43 2020
"""

#import get_tickers as get_tickers
import back_test as back_test
import pandas as pd
import sample_slopes as sample_slopes
import numpy as np
import settings as settings
import math

#batch=18
#look_ahead=2

tickers = ["MSFT", "GOOG","GOOGL", "FB", "T", "INTC",
           "VZ", "ADBE", "CSCO", "NVDA", "ORCL", "CRM",
           "ACN", "IBM", "AVGO", "TXN", "QCOM", "FIS",
           "TMUS", "INTU", "FISV", "ADP", "CCI",
           "NOW", "MU", "AMD", "AMAT", "ATVI", "LRCX","LHX"]

def performance_array(ticker,batch,look_ahead):
    """
    This test is used to try out the returns calculator on the stock market data for a single stock (change ticker)
    """
#    ticker='MSFT'
    
    main_df = pd.read_pickle(settings.settings_dict['stock_data_path'])
    main_df = sample_slopes.create_slope_sum_market(main_df)

    Back_Test = back_test.BackTest(main_df, settings.settings_dict['model_path'])

    y_values = sample_slopes.generate_target_values(main_df, batch, ticker + 'CLS', look_ahead)
    x_values = sample_slopes.create_batch_of_slopes(main_df, ticker + 'CLS', batch,   y_values[1])

    array_of_batches = Back_Test.create_batch_of_slopes(main_df, ticker + "slope_sum", batch, y_values[1])
    
    Back_Test.append_list_of_buy_sells(array_of_batches,ticker + "slope_sum")
    
    algorithm_return = Back_Test.take_bid_stream_calculate_profit(ticker + "bid_stream", batch, look_ahead, for_graph=True)
    
    array_of_bid_stream = Back_Test.main_df[ticker + 'bid_stream'].tolist()[batch + look_ahead - 1:] #1 or 0 long or short
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
    
    index_stock = list(range(0, len(main_df[ticker + 'CLS'].tolist()[batch + look_ahead - 1:])))

    stock_price=main_df[ticker + 'CLS'].tolist()[batch + look_ahead - 1:]
    
    stock_log_return_stream=Back_Test.test_calculate_holding_log_return(ticker + 'CLS')
    stock_log_return_stream=stock_log_return_stream[batch + look_ahead - 1:]
    algo_log_return_stream= Back_Test.take_bid_stream_calculate_log_return(ticker + "bid_stream", batch, look_ahead)
  
    return stock_log_return_stream,algo_log_return_stream


def performance_stream_callable(ticker,batch,look_ahead):
    """
    This test is used to try out the returns calculator on the stock market data for a single stock (change ticker)
    """
#    ticker='MSFT'
    
    # main_df = pd.read_pickle(settings.settings_dict['stock_data_path'])
    main_df = pd.read_pickle('data/df_without_NA.pkl')
    main_df = sample_slopes.create_slope_sum_market(main_df)

    Back_Test = back_test.BackTest(main_df, settings.settings_dict['model_path'])

    y_values = sample_slopes.generate_target_values(main_df, batch, ticker + 'CLS', look_ahead)
    x_values = sample_slopes.create_batch_of_slopes(main_df, ticker + 'CLS', batch,   y_values[1])

    array_of_batches = Back_Test.create_batch_of_slopes(main_df, ticker + "slope_sum", batch, y_values[1])
    
    Back_Test.append_list_of_buy_sells(array_of_batches,ticker + "slope_sum")
    
    algorithm_return = Back_Test.take_bid_stream_calculate_profit(ticker + "bid_stream", batch, look_ahead, for_graph=True)
    
    array_of_bid_stream = Back_Test.main_df[ticker + 'bid_stream'].tolist()[batch + look_ahead - 1:] #1 or 0 long or short
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
    
    index_stock = list(range(0, len(main_df[ticker + 'CLS'].tolist()[batch + look_ahead - 1:])))

    stock_price=main_df[ticker + 'CLS'].tolist()[batch + look_ahead - 1:]
    
    stock_log_return_stream=Back_Test.test_calculate_holding_log_return(ticker + 'CLS')
    stock_log_return_stream=stock_log_return_stream[batch + look_ahead - 1:]
    algo_log_return_stream= Back_Test.take_bid_stream_calculate_log_return(ticker + "bid_stream", batch, look_ahead)
    
    stock_log_return_stream=pd.DataFrame(stock_log_return_stream)
    algo_log_return_stream=pd.DataFrame(algo_log_return_stream)
    
    return stock_log_return_stream,algo_log_return_stream


if __name__ == '__main__':
    performance_array("MSFT",5,1)