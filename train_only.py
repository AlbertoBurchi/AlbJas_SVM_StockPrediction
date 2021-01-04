# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:50:29 2019

@author: alber
"""

import numpy as np
import time
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import csv

# import matplotlib as mpl
# from matplotlib import style
# from mpl_toolkits.mplot3d import Axes3D #MODAB
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm


# import stocks as st
from datetime import timedelta


import sample_slopes as sample_slopes
import support_vector as support_vector

plot_bool = 0

tickers = ["MSFT", "GOOG","GOOGL", "FB", "T", "INTC",
           "VZ", "ADBE", "CSCO", "NVDA", "ORCL", "CRM",
           "ACN", "IBM", "AVGO", "TXN", "QCOM", "FIS",
           "TMUS", "INTU", "FISV", "ADP", "CCI",
           "NOW", "MU", "AMD", "AMAT", "ATVI", "LRCX","LHX"]



start = dt.datetime(2015, 1, 1)
end = dt.datetime(2020, 8, 31) #end = dt.datetime(2019, 12, 31)

delta = timedelta(days=1)

batch_size = 7
look_ahead = 2


class Ticker_Data():
    """
    object to hold the stock data
    """

    def __init__(self, df):
        self.main_df = df

    def append_change_column(self, df, ticker):
        """
         take the dataframe that holds the stock info and the ticker of interest
         then append the new change column and the new close column to the main dataframe
        """
        df2 = pd.DataFrame()
        df2['change'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
        self.main_df[str(ticker) + 'CHG'] = df2['change']
        self.main_df[str(ticker) + 'CLS'] = df['Close']
        return self.main_df

    def backTester(self, df):
        for x in range(len(df.columns) - 2):
            df['stock' + str(x + 1)]
            df['stock1compair'] = np.where(df['stock' + str(x + 1)].values < df['stock' + str(x)].values and
                                           df['stock' + str(x + 1)].values < df['stock' + str(x + 2)].values, 1, 0)
        return df

    def drop_row_with_zeros(self):
        """
        removes the rows with zero on teh self.dataframe
        """

        columns = list(self.main_df)

        self.main_df = self.main_df[self.main_df[columns] != 0]

    def drop_row_with_NA(self):
        """
        removes the rows with NA on the self.dataframe
        """

        self.main_df = self.main_df.dropna()

def write_feature_and_targets(X, Y):
    """
    Takes the X and Y and writest them to a file
    """
    with open('data/all_feature.txt', 'w') as file_name:
        for feature in X:
            file_name.write(str(feature) + ',' + '\n')
        for target in Y:
            file_name.write(str(target) + ',' + '\n')


def main(batch_size, look_ahead):
    """
    da main function
    """
    i = 0
    main_df = pd.DataFrame()

    ticker_data = Ticker_Data(main_df)

    ticker_data.main_df = pd.read_pickle('data/df_without_NA_' + str(start).replace(':', '.') + '--' + str(end).replace(':', '.') + '.pkl')

    # add the slope sum values to the dataframe
    # ticker_data.main_df = sample_slopes.create_slope_sum(ticker_data.main_df)

    ticker_data.main_df = sample_slopes.create_slope_sum_market(ticker_data.main_df)

    # get the names of all the column titles
    columns = list(ticker_data.main_df)

    # get the names of the columns that have a slope_sum
    columns_with_sample_slopes = sample_slopes.get_columns_with_slope_sum(columns)

    # set up the ML package to hold the features and target values
    sv = support_vector.Support_Vector([], [])

    tdmdf=ticker_data.main_df

    for column in columns_with_sample_slopes:
        y_values = sample_slopes.generate_target_values(ticker_data.main_df, batch_size, column.replace('slope_sum', 'CLS'), look_ahead)
        # keeps adding new target values to varable
        sv.Y = sv.Y + y_values[0]
        # create_batch_of_slopes(df, batch_count, cut_length)
        # y_values[1] bec thats used to tell create batch_of_slopes where to
        # stop

        x_values = sample_slopes.create_batch_of_slopes(ticker_data.main_df, column, batch_size,   y_values[1])


        sv.X = sv.X + x_values

    write_feature_and_targets(sv.X, sv.Y)

#    print(sv.Y, 'Yvalues')
#    print(sv.X[-1], ' Xvalues ')

    # print sv.X, 'Xvalues'

    print('training the model...')

    precision, recall, f1_score, classification_reports, confustion_matrix, TruePositive, FalsePositive, FalseNegative, TrueNegative= sv.train()

    # sv.run_optunity()
#    prova=sv.predict_out_put (x_values)

#    for sample in test_data:
#    print (sv.predict_out_put([sample]))
    return ticker_data, columns,columns_with_sample_slopes,tdmdf,y_values,x_values, precision, recall, f1_score, classification_reports, confustion_matrix, TruePositive, FalsePositive, FalseNegative, TrueNegative

if '__main__' == __name__:
    ticker_data, columns,columns_with_sample_slopes,tdmdf,y_values,x_values, precision, recall, f1_score, classification_reports, confustion_matrix, TruePositive, FalsePositive, FalseNegative, TrueNegative=main(7,1)

