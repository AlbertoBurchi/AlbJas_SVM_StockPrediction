import get_tickers as get_tickers
import back_test as back_test
import pandas as pd
import sample_slopes as sample_slopes
import numpy as np
import settings
from performance_array import performance_stream_callable


def test_calculate_profit():
    """
    Makes sure that we can calculate the return if we just had held the stock
    """
    data = {
        'col4CLS': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.1, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4slope_sum': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.2, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4bid_stream': [None, None, None, None, None,1 , 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    stock_data = pd.DataFrame(data=data)

    print(len(data['col4CLS']))
    print(len(data['col4bid_stream']))

    # stock_data = pd.read_pickle('df_without_zeros.pkl')

    Back_Test = back_test.BackTest(
        stock_data, settings.settings_dict['test_model_path'])

    # def take_bid_stream_calculate_profit(self, column_bid_stream,
    # batch_size, look_ahead, for_graph=False):

    array_profit = Back_Test.take_bid_stream_calculate_profit(
        'col4bid_stream', 3, 2)
    print(len(array_profit))

    array_of_nones = []
    for i in range(len(stock_data['col4bid_stream'].index) - len(array_profit)):
        array_of_nones.append(None)

    print(len(stock_data['col4bid_stream'].index), ' len bid stream in df')
    print(len(array_of_nones + array_profit))

    stock_data['profit'] = array_of_nones + array_profit

    stock_data.to_csv('testing_files/test-data.csv')

    # needed to round the answers bec python and floats 
    rounded_profits = []
    for number in array_profit:
        rounded_profits.append(round(number,1))

    assert rounded_profits == [0,0.3 ,0.1,0,0,0,0,0.1,0.3,-0.3,0,0,0,0, -0.4,0.3,-0.7,0.4,0.4,0.1]


def test_calculate_profit_all_ones():
    """
    Makes sure that we can calculate the return if we just had held the stock
    """
    data = {
        'col4CLS': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.1, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4slope_sum': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.2, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4bid_stream': [None, None, None, None, None,1 , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    stock_data = pd.DataFrame(data=data)

    print(len(data['col4CLS']))
    print(len(data['col4bid_stream']))

    # stock_data = pd.read_pickle('df_without_zeros.pkl')

    Back_Test = back_test.BackTest(
        stock_data, settings.settings_dict['test_model_path'])

    # def take_bid_stream_calculate_profit(self, column_bid_stream,
    # batch_size, look_ahead, for_graph=False):

    array_profit = Back_Test.take_bid_stream_calculate_profit('col4bid_stream', 3, 2)
    print(len(array_profit))

    array_of_nones = []
    for i in range(len(stock_data['col4bid_stream'].index) - len(array_profit)):
        array_of_nones.append(None)

    print(len(stock_data['col4bid_stream'].index), ' len bid stream in df')
    print(len(array_of_nones + array_profit))

    stock_data['profit'] = array_of_nones + array_profit

    stock_data.to_csv('testing_files/test-data.csv')

    # needed to round the answers bec python and floats 
    rounded_profits = []
    for number in array_profit:
        rounded_profits.append(round(number,1))

    assert rounded_profits == [0,0.3 ,0.1,0.1,0.1,0.1,0.1,0.1,0.3,-0.3,0.5,0.1,-0.6,-0.1,-0.4,0.3,-0.7,0.4,0.4,0.1]
    
def test_calculate_profit_all__minus_ones():
    """
    Makes sure that we can calculate the return if we just had held the stock
    """
    data = {
        'col4CLS': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.1, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4slope_sum': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.2, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4bid_stream': [None, None, None, None, None,-1 , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    }
    stock_data = pd.DataFrame(data=data)

    print(len(data['col4CLS']))
    print(len(data['col4bid_stream']))

    # stock_data = pd.read_pickle('df_without_zeros.pkl')

    Back_Test = back_test.BackTest(
        stock_data, settings.settings_dict['test_model_path'])

    # def take_bid_stream_calculate_profit(self, column_bid_stream,
    # batch_size, look_ahead, for_graph=False):

    array_profit = Back_Test.take_bid_stream_calculate_profit('col4bid_stream', 3, 2)
    print(len(array_profit))

    array_of_nones = []
    for i in range(len(stock_data['col4bid_stream'].index) - len(array_profit)):
        array_of_nones.append(None)

    print(len(stock_data['col4bid_stream'].index), ' len bid stream in df')
    print(len(array_of_nones + array_profit))

    stock_data['profit'] = array_of_nones + array_profit

    stock_data.to_csv('testing_files/test-data.csv')

    # needed to round the answers bec python and floats 
    rounded_profits = []
    for number in array_profit:
        rounded_profits.append(round(number,1))

#    assert rounded_profits == [0,0.3 ,0.1,0.1,0.1,0.1,0.1,0.1,0.3,-0.3,0.5,0.1,-0.6,-0.1,-0.4,0.3,-0.7,0.4,0.4,0.1]
    assert rounded_profits == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]    
    
def test_calculate_profit_bid_start_short():
    """
    Makes sure that we can calculate the return if we just had held the stock
    """
    data = {
        'col4CLS': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.1, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4slope_sum': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.2, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4bid_stream': [None, None, None, None, None,-1 , -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    stock_data = pd.DataFrame(data=data)

    print(len(data['col4CLS']))
    print(len(data['col4bid_stream']))

    # stock_data = pd.read_pickle('df_without_zeros.pkl')

    Back_Test = back_test.BackTest(
        stock_data, settings.settings_dict['test_model_path'])

    # def take_bid_stream_calculate_profit(self, column_bid_stream,
    # batch_size, look_ahead, for_graph=False):

    array_profit = Back_Test.take_bid_stream_calculate_profit(
        'col4bid_stream', 3, 2)
    print(len(array_profit))

    array_of_nones = []
    for i in range(len(stock_data['col4bid_stream'].index) - len(array_profit)):
        array_of_nones.append(None)

    print(len(stock_data['col4bid_stream'].index), ' len bid stream in df')
    print(len(array_of_nones + array_profit))

    stock_data['profit'] = array_of_nones + array_profit

    stock_data.to_csv('testing_files/test-data.csv')

    # needed to round the answers bec python and floats 
    rounded_profits = []
    for number in array_profit:
        rounded_profits.append(round(number,1))

    assert rounded_profits == [0,0,0,0,0,0,0,0.1,0.3,-0.3,0,0,0,0, -0.4,0.3,-0.7,0.4,0.4,0.1]

def test_calculate_profit_bid_start_short_for_graph():
    """
    Makes sure that we can calculate the return if we just had held the stock
    """
    data = {
        'col4CLS': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.1, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4slope_sum': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.2, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4bid_stream': [None, None, None, None, None,-1 , -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    stock_data = pd.DataFrame(data=data)
    # stock_data.to_csv('files/test caculate profit mock data')

    print(len(data['col4CLS']))
    print(len(data['col4bid_stream']))

    # stock_data = pd.read_pickle('df_without_zeros.pkl')

    Back_Test = back_test.BackTest(
        stock_data, settings.settings_dict['test_model_path_jason'])

    # def take_bid_stream_calculate_profit(self, column_bid_stream,
    # batch_size, look_ahead, for_graph=False):

    array_profit = Back_Test.take_bid_stream_calculate_profit_jason2020(
        'col4bid_stream', 3, 2,  for_graph=True)
    print(len(array_profit))

    array_of_nones = []
    for i in range(len(stock_data['col4bid_stream'].index) - len(array_profit)):
        array_of_nones.append(None)

    print(len(stock_data['col4bid_stream'].index), ' len bid stream in df')
    print(len(array_of_nones + array_profit))

    stock_data['profit'] = array_of_nones + array_profit

    stock_data.to_csv('testing_files/test-data.csv')

    # needed to round the answers bec python and floats 
    rounded_profits = []
    for number in array_profit:
        rounded_profits.append(round(number,1))

    assert rounded_profits == [0,0,0,0,0,0,0,0.1,0.3,-0.3,0,0,0,0, -0.4,0.3,-0.7,0.4,0.4,0.1]
    
def test_calculate_profit_bid_start_long_for_graph():
    """
    Makes sure that we can calculate the return if we just had held the stock
    """
    data = {
        'col4CLS': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.1, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4slope_sum': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.2, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4bid_stream': [None, None, None, None, None,1 , 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    stock_data = pd.DataFrame(data=data)

    print(len(data['col4CLS']))
    print(len(data['col4bid_stream']))

    # stock_data = pd.read_pickle('df_without_zeros.pkl')

    Back_Test = back_test.BackTest(
        stock_data, settings.settings_dict['test_model_path'])

    # def take_bid_stream_calculate_profit(self, column_bid_stream,
    # batch_size, look_ahead, for_graph=False):

    array_profit = Back_Test.take_bid_stream_calculate_profit(
        'col4bid_stream', 3, 2, for_graph=True)
    print(len(array_profit))

    array_of_nones = []
    for i in range(len(stock_data['col4bid_stream'].index) - len(array_profit)):
        array_of_nones.append(None)

    print(len(stock_data['col4bid_stream'].index), ' len bid stream in df')
    print(len(array_of_nones + array_profit))

    stock_data['profit'] = array_of_nones + array_profit

    stock_data.to_csv('testing_files/test-data.csv')

    # needed to round the answers bec python and floats 
    rounded_profits = []
    for number in array_profit:
        rounded_profits.append(round(number,1))

#    assert rounded_profits == [0,0.3 ,0.1,0,0,0,0,0.1,0.3,-0.3,0,0,0,0, -0.4,0.3,-0.7,0.4,0.4,0.1]

def test_calculate_holding_log_return():
    """
    Makes sure that we can calculate the return if we just had held the stock
    """
    data = {
        'col4CLS': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.1, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4slope_sum': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.2, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4bid_stream': [None, None, None, None, None,1 , 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    stock_data = pd.DataFrame(data=data)

    print(len(data['col4CLS']))
    print(len(data['col4bid_stream']))

    # stock_data = pd.read_pickle('df_without_zeros.pkl')

    Back_Test = back_test.BackTest(
        stock_data, settings.settings_dict['test_model_path'])

    # def take_bid_stream_calculate_profit(self, column_bid_stream,
    # batch_size, look_ahead, for_graph=False):

    holding_logreturn = Back_Test.calculate_holding_logreturn('col4CLS', 3, 2)
    holding_logreturn = Back_Test.calculate_holding_logreturn('col4CLS')


    # needed to round the answers bec python and floats 
    rounded_holding_logreturn = []
    rounded_holding_logreturn.append(round(holding_logreturn,2))

    assert rounded_holding_logreturn == [0.56]
    
def test_take_bid_stream_calculate_log_return():
    """
    Makes sure that we can calculate the return if we just had held the stock
    """
    data = {
        'col4CLS': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.1, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4slope_sum': [1, 1.2, 1.3, 1.7, 1.5, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.4, 2.2, 2.6, 2.7, 2.1, 2.0, 1.6, 1.9, 1.2, 1.6, 2, 2.1],
        'col4bid_stream': [None, None, None, None, None,1 , 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    stock_data = pd.DataFrame(data=data)

    print(len(data['col4CLS']))
    print(len(data['col4bid_stream']))

    # stock_data = pd.read_pickle('df_without_zeros.pkl')

    Back_Test = back_test.BackTest(
        stock_data, settings.settings_dict['test_model_path'])

    # def take_bid_stream_calculate_profit(self, column_bid_stream,
    # batch_size, look_ahead, for_graph=False):

    array_algo_logreturn = Back_Test.take_bid_stream_calculate_log_return('col4bid_stream', 3, 2)
    print(len(array_algo_logreturn))

    array_of_nones = []
    for i in range(len(stock_data['col4bid_stream'].index) - len(array_algo_logreturn)):
        array_of_nones.append(None)

    print(len(stock_data['col4bid_stream'].index), ' len bid stream in df')
    print(len(array_of_nones + array_algo_logreturn))

    stock_data['profit'] = array_of_nones + array_algo_logreturn

    stock_data.to_csv('testing_files/test-data.csv')
    sum_algo_logreturn=sum(array_algo_logreturn)
    
        # needed to round the answers bec python and floats
    rounded_algo_logreturn = []
    rounded_algo_logreturn.append(round(sum_algo_logreturn,2))
    
    assert rounded_algo_logreturn == [0.39]


    # assert rounded_profits == [0,0.22,0.6,0,0,0,0,0.05,0.13,-0.13,0,0,0,0,-0.22,0.17,-0.46,0.29,0.22,0.05,0.39]

    # assert rounded_profits == [0,0.22,0.06,0,0,0.00,0.00,0.05,0.13,-0.13,0.00,0.00,0.00,0.00,-0.22,0.17,-0.46,0.29,0.22,0.05,0.39]
    
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
    stock_log_return_stream=stock_log_return_stream[3 + 2 - 1:]
    algo_log_return_stream= Back_Test.take_bid_stream_calculate_log_return("col4bid_stream", 3, 2)
  
