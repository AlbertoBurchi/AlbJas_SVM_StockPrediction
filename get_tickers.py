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
from mpl_toolkits.mplot3d import Axes3D #MODAB
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
# tickers = ["MSFT", "GOOG","GOOGL", "FB", "T"]


start = dt.datetime(2015, 1, 1)
end = dt.datetime(2020, 8, 31) #end = dt.datetime(2019, 12, 31)

delta = timedelta(days=1)

batch_size = 18
look_ahead = 2

#plot_df = pd.DataFrame()
#df2 = pd.DataFrame()
#main_df = pd.DataFrame()

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

    def create3d_plot(self):

        change_columns = [col for col in self.main_df.columns if 'CHG' in col]

        print (change_columns)
        new_df_with_change_columns = self.main_df[change_columns].copy()
        print (new_df_with_change_columns)

        tickers_without_change_label  = []
        for chg_column in change_columns:
            tickers_without_change_label.append(chg_column.replace("CHG",""))


        # for CHG_column in new_df_with_change_columns:
        #     print (CHG_column)

        x = np.arange(len(new_df_with_change_columns.columns))

        y = new_df_with_change_columns.index
        X,Y = np.meshgrid(x,y)
        Z = new_df_with_change_columns
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = fig.add_subplot(111, projection='3d')
        # ax.auto_scale_xyz([0, 7], [0, 2000], [-.1, .1])

        locs, labels=plt.xticks()

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('Stocks Sorted by Market Cap')
        ax.set_ylabel('Days (Starting Jan 1, 2015)')
        ax.set_zlabel('Percent Change Per Day')

        plt.show()


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
#     NOTE ============start here to get new stock data=======================
    for ticker in tickers:

        print (ticker)
        time.sleep(.02)
        #df = web.DataReader(ticker, 'iex', start, end)
        df = web.DataReader(ticker, 'yahoo', start, end)
        df = df.reset_index(level='Date')
        print (df.head())
        ticker_data.append_change_column(df, ticker)
        i = i + 50
    # remove the rows that contain any 0's or NA
    # ticker_data.main_df.to_csv('stock_data/before_NA_drop_stock_data_slope_sumNoNA' + str(start).replace(':', '.') + '--' + str(end).replace(':', '.') + '.csv')
    # ticker_data.drop_row_with_zeros()
    ticker_data.drop_row_with_NA()
#    ticker_data.dropna()
    #ticker_data.main_df.to_pickle('stock_data/df_without_NA_' + str(start) + '--' + str(end) + '.pkl')
    ticker_data.main_df.to_pickle('data/df_without_NA_' + str(start).replace(':', '.') + '--' + str(end).replace(':', '.') + '.pkl')
    ticker_data.main_df.to_pickle('data/df_without_NA.pkl')
#     NOTE ============end================================================


    ticker_data.create3d_plot()



    ticker_data.main_df = pd.read_pickle('data/df_without_NA_' + str(start).replace(':', '.') + '--' + str(end).replace(':', '.') + '.pkl')

    # add the slope sum values to the dataframe
    # ticker_data.main_df = sample_slopes.create_slope_sum(ticker_data.main_df)

    ticker_data.main_df = sample_slopes.create_slope_sum_market(ticker_data.main_df)

    # write the whole datarame to a csv if you want to
    # ticker_data.main_df.to_csv('stock_data/stock_data_slope_sumNoNA' + str(start).replace(':', '.') + '--' + str(end).replace(':', '.') + '.csv')

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

        # x_values = sample_slopes.create_batch_of_slopes_moving_av(
        #     ticker_data.main_df, column, batch_size,   y_values[1], 15)

        # keeps adding new feature values to varable
        sv.X = sv.X + x_values

    write_feature_and_targets(sv.X, sv.Y)

#    print(sv.Y, 'Yvalues')
#    print(sv.X[-1], ' Xvalues ')

    # print sv.X, 'Xvalues'

    print('training the model...')

    precision, recall, f1_score, classification_reports, confustion_matrix, TruePositive, FalsePositive, FalseNegative, TrueNegative= sv.train()

#    print("running optunity")
#    sv.run_optunity()
#    prova=sv.predict_out_put (x_values)

#    for sample in test_data:
#    print (sv.predict_out_put([sample]))

    # return ticker_data.main_df, columns,columns_with_sample_slopes,tdmdf,y_values,x_values



    return precision, recall, f1_score

def iterate_over_all_batch_and_look_ahead():
    """
    iterates over all the differnt comindations and writes to a csv file
    """

    # for j in range(3, 15):
    #     for i in range(2, 50):
    for j in range(1, 5):
        for i in range(13, 19):
#            sell, buy = main(i, j)
            precision, recall, f1_score = main(i, j)

            # with open('files/testing_files/hold_percents.txt', 'a')as f:
            #     f.write(str(float(sell[0]) / float((sell[1] + sell[0]))) + ',' +
            #             str(float(buy[1]) / float(buy[1] + buy[0])) +
            #             ', ' + str(i) + ',' + str(j) + '\n')

            with open('files/testing_files/hold_percents_jan2020.csv', mode='a') as target_file:
                    target_files = csv.writer(
                        target_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    target_files.writerow([precision, recall, f1_score,i,j])

            print(i, j)


if '__main__' == __name__:

#    sell, buy = main(18, 2)

#    main_df,columns,columns_with_sample_slopes,tdmdf,y_values,x_values=main(18, 2)

    main(18, 2)
    # iterate_over_all_batch_and_look_ahead()

    # start = dt.datetime(2012, 10, 1)
    # end = dt.datetime(2018, 4, 14)
    # ticker = 'HPQ'
    # df = web.DataReader(ticker, 'iex', start, end)
    # print df
    # i = 0
    # arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # while i < (len(arr) - 2 - 1):
    #     print i
    #     i += 1
    # print arr[i], 'hree is '
    # print i



#dfchange = main_df.filter(regex='CHG')
#dfchange = dfchange.dropna() #remove NA
#
#print (dfchange)
#
#xpos=np.arange(dfchange.shape[0])
#ypos=np.arange(dfchange.shape[1])
##yposM, xposM = np.meshgrid(ypos+0.5, xpos+0.5)
#yposM, xposM = np.meshgrid(ypos, xpos)
#zpos=np.zeros(dfchange.shape).flatten()
#
#
#dx = 0.5 * np.ones_like(zpos)
#dy= 0.1 * np.ones_like(zpos)
#dz=dfchange.values.ravel()
#
#fig = plt.figure(figsize=(12,12))
#ax = fig.add_subplot(111, projection='3d')
#
#values = np.linspace(0.2, 1., xposM.ravel().shape[0])
##colors = cm.rainbow(values)
#ax.bar3d(xposM.ravel(),yposM.ravel(),zpos,dx,dy,dz, alpha=0.5)
