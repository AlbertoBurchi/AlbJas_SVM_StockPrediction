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

nticker = 30 

def log_return_on_market_data_single_stock(ticker,batch,look_ahead):
    """
    This test is used to try out the returns calculator on the stock market data for a single stock (change ticker)
    """
#    ticker='QCOM'

    main_df = pd.read_pickle(settings.settings_dict['stock_data_path'])
    main_df = sample_slopes.create_slope_sum(main_df)

    Back_Test = back_test.BackTest(main_df, settings.settings_dict['model_path'])

    y_values = sample_slopes.generate_target_values(main_df, batch, ticker + 'CLS', look_ahead)
    x_values = sample_slopes.create_batch_of_slopes(main_df, ticker + 'CLS', batch,   y_values[1])

    array_of_batches = Back_Test.create_batch_of_slopes(main_df, ticker + "slope_sum", batch, y_values[1])

    print(array_of_batches, ' here is lensss')

    print(Back_Test.append_list_of_buy_sells(array_of_batches,ticker + "slope_sum"))

    print("algorithm log return", sum(Back_Test.take_bid_stream_calculate_log_return(ticker + "bid_stream", batch, look_ahead)) * 100, '%')
    print("log return ", sum(Back_Test.test_calculate_holding_log_return(ticker + 'CLS')) * 100, '%')
    print("percent change", Back_Test.calculate_holding_percent_change_return(ticker + 'CLS') * 100, '%')

def performance(batch,look_ahead): #TODO
    """
    TO DO
    """
    
    """
    This test is used to try out the returns calculator on the stock market data
    """
    #"SHOP", "DELL", "UBER",

    main_df = pd.read_pickle(settings.settings_dict['stock_data_path'])
    main_df = sample_slopes.create_slope_sum_market(main_df)

    Back_Test = back_test.BackTest(main_df, settings.settings_dict['model_path'])

    with open('results/return_single_stock.csv', 'w') as f:
        mean_array = []
        std_array = []
        returns_difference_array = []
        holding_final_capital_array = []
        algo_final_capital_array = []
        algo_aggregation_cross_section_ptf =[]
        for ticker in tickers:

            slope_sums = main_df[ticker + "slope_sum"]

            mean = np.mean(main_df[ticker + "slope_sum"])
            std = np.std(main_df[ticker + "slope_sum"])

            y_values = sample_slopes.generate_target_values(main_df, batch, ticker + "CLS", look_ahead)
            x_values = sample_slopes.create_batch_of_slopes(main_df, ticker + 'CLS', batch,   y_values[1])

            array_of_batches = Back_Test.create_batch_of_slopes(main_df, ticker + 'slope_sum', batch, y_values[1])

            Back_Test.append_list_of_buy_sells(array_of_batches,ticker + "slope_sum")

            algorithm_profit = sum(Back_Test.take_bid_stream_calculate_profit(ticker + "bid_stream", batch, look_ahead))
            log_return = sum(Back_Test.test_calculate_holding_log_return(ticker + 'CLS'))
            holding_profit = Back_Test.calculate_holding_profit(ticker + "CLS", batch, look_ahead)

            print ("Ticker", ticker)
            print("algorithm profit", algorithm_profit)
            print("log return ", log_return, ' %')
            print("holding profit", holding_profit, end=' ')

            algorithm_log_return = sum(Back_Test.take_bid_stream_calculate_log_return(ticker + "bid_stream", batch, look_ahead)) * 100
            log_return = sum(Back_Test.test_calculate_holding_log_return(ticker + 'CLS')) * 100
            percent_change = Back_Test.calculate_holding_percent_change_return(ticker + "CLS") * 100

            print ("Ticker", ticker)
            print("algorithm log return", algorithm_log_return, '%')
            print("log return ", log_return, ' %')
            print("percent change", percent_change, '%')


            mean_array.append(mean)
            std_array.append(std)
            returns_difference_array.append(algorithm_profit - holding_profit)

            f.write(ticker + ',' + str(algorithm_log_return) + ',' + str(log_return) +
                    ',' + str(percent_change) + ','+
                    str(algorithm_profit) + ',' + str(log_return) +
                    ',' + str(holding_profit) + ',' + str(mean) + ',' + str(std) + '\n')
            
            holding_final_capital=100*(1+percent_change/100)
            algo_aggregation = 1/nticker*math.exp(algorithm_log_return/100)
                  
            
            holding_final_capital_array.append(holding_final_capital)
            algo_final_capital_array.append(algo_aggregation)
                       

#        data = {
#            'mean': mean_array,
#            'std': std_array,
#            'returns_diff': returns_difference_array
#        }
#        meaningfull_stats = pd.DataFrame(data=data)
#
#        meaningfull_stats.to_pickle('files/meaningfull_stats.pkl')
#        
        ptf_holding_final_capital=sum(holding_final_capital_array)/nticker
        ptf_algo_aggregation=math.log(sum(algo_final_capital_array))
            
        ptf_algo_final_capital=100*math.exp(ptf_algo_aggregation)
        
#
#        return meaningfull_stats.to_pickle('files/meaningfull_stats.pkl')

        return ptf_holding_final_capital, ptf_algo_final_capital

def ptfvalue(batch,look_ahead): #TODO
    """
    TO DO
    """
    
    """
    This test is used to try out the returns calculator on the stock market data
    """
    #"SHOP", "DELL", "UBER",

    main_df = pd.read_pickle(settings.settings_dict['stock_data_path'])
    main_df = sample_slopes.create_slope_sum_market(main_df)

    Back_Test = back_test.BackTest(main_df, settings.settings_dict['model_path'])

    with open('results/statistics_single_stock.csv', 'w') as f:
        mean_array = []
        std_array = []
        logret_difference = []
        holding_final_capital_array = []
        algo_final_capital_array = []
                
        for ticker in tickers:

            slope_sums = main_df[ticker + "slope_sum"]

            mean = np.mean(main_df[ticker + "slope_sum"])
            std = np.std(main_df[ticker + "slope_sum"])

            y_values = sample_slopes.generate_target_values(main_df, batch, ticker + "CLS", look_ahead)
            x_values = sample_slopes.create_batch_of_slopes(main_df, ticker + 'CLS', batch,   y_values[1])

            array_of_batches = Back_Test.create_batch_of_slopes(main_df, ticker + 'slope_sum', batch, y_values[1])

            Back_Test.append_list_of_buy_sells(array_of_batches,ticker + "slope_sum")
            
                        
            holding_logreturn = Back_Test.calculate_holding_logreturn(ticker + 'CLS', batch, look_ahead)
            array_algo_logreturn = Back_Test.take_bid_stream_calculate_log_return(ticker + "bid_stream", batch, look_ahead)
            
            # adjust for the correct lenght
            # array_of_nones = []
            # for i in range(len(ticker + "bid_stream".index) - len(array_algo_logreturn)):
            #     array_of_nones.append(None)

            # array_algo_logreturn_correct_lenght = array_of_nones + array_algo_logreturn

            algo_logreturn=sum(array_algo_logreturn)
            holding_final_capital=100*math.exp(holding_logreturn)
            algo_final_capital=100*math.exp(algo_logreturn)
            
            
            mean_array.append(mean)
            std_array.append(std)
            logret_difference=algo_logreturn - holding_logreturn
            
            f.write(ticker + ',' + str(algo_logreturn) + ',' + str(holding_logreturn) +
                    ',' + str(logret_difference) +
                    ',' +  str(algo_final_capital) + ',' + str(holding_final_capital) +
                    ',' + str(mean) + ',' + str(std) + '\n')                  
            
            holding_final_capital_array.append(holding_final_capital)
            algo_final_capital_array.append(algo_final_capital)
                       

#        
        ptf_holding_final_capital=sum(holding_final_capital_array)
        ptf_algo_final_capital=sum(algo_final_capital_array)
            
#
#        return meaningfull_stats.to_pickle('files/meaningfull_stats.pkl')

        return ptf_holding_final_capital, ptf_algo_final_capital

if __name__ == '__main__':
    log_return_on_market_data_single_stock('GOOG', 5, 1)
#test_on_market_data_single_stock()
#test_performance()
