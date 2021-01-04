# Steps
1.	#Main file: get stock data; train a new model; make beautiful plot;
get_tickers.py
initial set is 18,2

2.	#Train the model without download new tickers and data (just using data saved before).
train_only.py
not required. allows you to test a single model
initial set is 7,1

3.	# Train different models changing batch size and look_ahead
Iteration
Output 2 different files (csv)
return_single_stock.csv
	performance for each stock in the basket for each iteration setting
	in columns hlding vs algo profit for different ij settings
iteration_performance.csv
	columns: precision, recall, f1_score,TruePositive, FalsePositive, FalseNegative, TrueNegative,ptf_holding_final_capital, ptf_algo_final_capital,i,j (we need to add columns labels… but now I’m not able)

4.	#train the model then calculates and builds a chart and csv file with the vectors of the daily yields and the capital value of the strategies compared: buy and hold vs algorithm. Initial investment is 100 fo each stock (no portfolio optimization or risk management)
train_only_7_1_with_performance_stream
initial set is 7,1

Output 2 different files (xls)
	stock_log_return_stream_all
algo_log_return_stream_all






Support files
support_vector
#Contains modules for the Support Vector Machine optimization procedure
sample_slope
# Contains modules for creating the "sample slope" indicator. It consists in the difference between the returns of the single stock with the returns of all the other stocks contained in the basket
back_test
# it contains all the modules necessary to evaluate the model
return_calculator
# contains the modules that allow you to measure the performance of the algorithm
#Calculate the return of the algoritm and the buy-and-hold strategy
#also other unused  methods
settings
#support file for the "iteration" procedure and ... add more ...
iteration_only.py

#Plot the backtest of the model on one selected ticker/stock
plot_stock.py
