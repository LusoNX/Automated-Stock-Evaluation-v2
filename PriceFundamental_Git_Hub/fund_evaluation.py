from datetime import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
import time
import requests
import sqlite3
from selenium import webdriver  
from bs4 import BeautifulSoup
import random
from scipy import stats
from scipy.stats import norm
from pandas.io.json import json_normalize
import requests
import urllib
from sqlalchemy import create_engine
import pyodbc
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import random
from datetime import date
import investpy
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

import operator
conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=C:\Users\35196\OneDrive\Desktop\Investment Prospectus\DataBase\Funds\FundDataBase.accdb;')
cnn_url = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
acc_engine = create_engine(cnn_url)

def get_matrix_data(id_index,id_benchmark):

	# id index is the industry in which we want to operate, while id_benchmark is the benchmark (market as the base case)
	matrix_directory = r"C:\Users\35196\OneDrive\Desktop\Investment Prospectus\DataBase\Funds\Matrix Figures"
	today_date = date.today()
	beg_date =  date.today() - relativedelta(years=8)
	today_date = np.datetime64(today_date)
	beg_date = np.datetime64(beg_date)

	if id_index == id_benchmark:
		pass
	else:
		df_price =pd.read_sql("SELECT * FROM IndexPriceData WHERE ID_INDEX ={}".format(id_index),acc_engine,index_col = "Data").sort_index()
		df_benchmark = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX ={}".format(id_benchmark),acc_engine,index_col = "Data").sort_index()
		df_benchmark.rename(columns = {"Close":"Close_market"},inplace = True)
		df_benchmark = df_benchmark[["Close_market"]]

		date_range = pd.date_range(beg_date, today_date, freq='D')
		df_returns_all = pd.DataFrame(index = date_range)
		df_price_y = df_price.resample("Y").last()
		nr_years = len(df_price_y)
		id_name = pd.read_sql("SELECT name FROM IndexData WHERE ID_INDEX = {}".format(id_index),acc_engine).values[0][0]
		id_name = "Close_{}".format(id_name)
		df_price[id_name] = df_price["Close"].pct_change()
		df_price.dropna(inplace = True)
		if nr_years >= 3 :
			df_new = df_price[[id_name]]
			df_returns_all = df_new.merge(df_returns_all,how = "inner", right_index = True, left_index = True)
		else:
			pass
		
		corr_matrix_all = df_returns_all.corr().round(2)
		fig, ax = plt.subplots(figsize=(20,8))
		ax = sns.heatmap(corr_matrix_all, annot=True)
		plt.tight_layout()
		ax.set_title("Correlation Matrix among Funds")
		ax.set_xlabel("Stock ID")
		ax.set_ylabel("Stock ID")
		#plt.show()
		fig.savefig(matrix_directory+r"\corr_matrix_{}.png".format(today_date),dpi = 300)
		plt.close()

		## ADD space for corr matrix by industry
		def coskew(df, bias = False):
			v = df.values
			s1 = sigma = v.std(0,keepdims = True)
			means = v.mean(0,keepdims = True)

			v1 = v - means
			s2 = sigma ** 2
			v2 = v1 ** 2
			m = v.shape[0]
			skew = pd.DataFrame(v2.T.dot(v1) / s2.T.dot(s1) / m, df.columns, df.columns)
			if not bias:
				skew *= ((m - 1) * m) ** .5 / (m - 2)

			return skew

		df_coskew_all = coskew(df_returns_all).round(3)
		plt.figure(figsize =(15,8))
		sns.heatmap(df_coskew_all, annot=True)
		plt.tight_layout()
		plt.title("Coskewness Among Stock")
		plt.xlabel("Stock Fund ID")
		plt.ylabel("Stock Fund ID")
		#plt.show()
		#plt.savefig(matrix_directory+r"\skew_matrix_{}.png".format(today_date),dpi = 300)
		plt.close()



		def cokurt(df, bias=False, fisher=True, variant='middle'):
			v = df.values
			s1 = sigma = v.std(0, keepdims=True)
			means = v.mean(0, keepdims=True)
			# means is 1 x n (n is number of columns
			# this difference broacasts appropriately
			v1 = v - means
			s2 = sigma ** 2
			s3 = sigma ** 3
			v2 = v1 ** 2
			v3 = v1 ** 3
			m = v.shape[0]

			if variant in ['left', 'right']:
				kurt = pd.DataFrame(v3.T.dot(v1) / s3.T.dot(s1) / m, df.columns, df.columns)
				if variant == 'right':
					kurt = kurt.T
			elif variant == 'middle':
				kurt = pd.DataFrame(v2.T.dot(v2) / s2.T.dot(s2) / m, df.columns, df.columns)
			if not bias:
				kurt = kurt * (m ** 2 - 1) / (m - 2) / (m - 3) - 3 * (m - 1) ** 2 / (m - 2) / (m - 3)
			if not fisher:
				kurt += 3
			return kurt

		df_cokur = cokurt(df_returns_all).round(2)
		plt.figure(figsize = (15,8))
		sns.heatmap(df_cokur, annot=True)
		plt.tight_layout()
		plt.title("Cokurtosis Among Stock")
		plt.xlabel("Mutual Fund ID")
		plt.ylabel("Mutual Fund ID")
		plt.savefig(matrix_directory+r"\kurt_matrix_{}.png".format(today_date))
		plt.close()


		## Get Cosk for market data
		df_price = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {} ".format(id_index),acc_engine,index_col = "Data").sort_values("Data")
		df_stock = df_price.merge(df_benchmark,how = "inner", right_index = True, left_index = True)

		# METHOD 1, Uses the methodology from Kraus Litzig () where they use the unconditional standardized coskweness   by dividing coskew with the market by the skewness of the market	
		df_stock = df_stock.resample("M").last()
		df_stock = df_stock.pct_change()
		df_stock = df_stock.dropna()		
		new_beg_date =  date.today() - relativedelta(years=8) # Select new beg date
		new_beg_date = np.datetime64(new_beg_date)
		mask = (df_stock.index > new_beg_date) & (df_stock.index <= today_date)
		df_stock = df_stock[mask]
		len_stock = len(df_stock)
		range_len = 36 # range of betas is 36 ( 3 years)
		regression_period = 60 # Periodo de regressão de cada beta é de 5 anos
		index_values = df_stock[-range_len:].index # Betas for 5 years

		corr_list = []
		for z in range(0,range_len):
			df_new_stock =df_stock.iloc[0+z:(regression_period+z)] # Betas for 5 years
			corr_market = df_new_stock.corr().round(2)
			corr_values = corr_market["Close"].iloc[-1]
			corr_list.append(corr_values)

		coskew_list = []
		unconditional_cos_list = []
		for z in range(0,range_len):
			df_new_stock =df_stock.iloc[0+z:(regression_period+z)] # Betas for 5 years
			df_new_coskew = coskew(df_new_stock)
			skew_market = df_new_stock.Close_market.skew()
			cos_values = df_new_coskew["Close"].iloc[-1]
			unconditional_cos =  cos_values/skew_market
			coskew_list.append(cos_values)
			unconditional_cos_list.append(unconditional_cos)

			
		# METHOD 2 , USES THE METHODOLOGY OF Moreno and Rodrigez (), where coskew with the market, is defined by the quadratic regression of the stok returns over the market returns. 
		quadratic_coskew_list = []
		for z in range(0,range_len):
			df_new_stock =df_stock.iloc[0+z:(regression_period+z)] # Betas for 5 years
			X = np.c_[df_new_stock[["Close_market"]]]
			Y = np.c_[df_new_stock["Close"]]
			polynomial_features = PolynomialFeatures(degree=2)
			X = polynomial_features.fit_transform(X)
			lin_reg_model = sklearn.linear_model.LinearRegression().fit(X,Y)
			beta_1 = lin_reg_model.coef_[0][0]
			beta_2 = lin_reg_model.coef_[0][1]
			est = sm.OLS(Y, X)
			est = est.fit()
			quadratic_coskew_list.append(beta_2)

		df_coskew_stock = pd.DataFrame([coskew_list]).T

		df_coskew_stock.rename(columns={0:"coskew_bench"},inplace = True)
		df_coskew_stock["unconditional_cos"] = unconditional_cos_list
		df_coskew_stock["quadratic_cos"] = quadratic_coskew_list
		df_coskew_stock["corr_benchmark"] = corr_list
		df_coskew_stock.index = index_values

		# METHOD 3, USES THE METHODOLOGY OF Harvey and Siddique where standardized coskewnewss is defined as the residual regression of asset on market return, divided by the dot product of residual market over its mean
		#["Use the same logic for Harvey"]
		last_coskew_market = df_coskew_stock.iloc[[-1]]
		plt.figure(figsize = (10,6))
		sns.heatmap(last_coskew_market, annot=True)
		plt.tight_layout()
		plt.title("Coskewness with Market")
		plt.xlabel("Stock Fund ID")
		plt.ylabel("Stock Fund ID")
		#plt.show()
		#plt.savefig(matrix_directory+r"\kurt_matrix_{}.png".format(today_date))
		plt.close()


		df_coskew_stock["ID_INDEX"] = id_index
		df_coskew_stock["ID_BENCH"] =id_benchmark
		df_coskew_stock.reset_index(inplace = True)
		df_coskew_stock.set_index("ID_INDEX",inplace = True)
		df_coskew_stock = df_coskew_stock.sort_values("Data")
		df_coskew_stock_exists = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX= {} AND ID_BENCH = {}".format(id_index,id_benchmark),acc_engine).sort_values("Data")

		if df_coskew_stock_exists.empty:
			df_coskew_stock.to_sql("CoSkewnessKurt",acc_engine,if_exists = "append")
		else:
			last_date = df_coskew_stock_exists["Data"].iloc[-1]
			last_date =np.datetime64(last_date)
			mask_2  = (df_coskew_stock["Data"] > last_date)
			df_coskew_stock = df_coskew_stock[mask_2]
			df_coskew_stock.to_sql("CoSkewnessKurt",acc_engine,if_exists = "append")







def clustering_classification_skew(get_data):
	# Use the 1st 10, excluding 3, for a representation of all sector/industry in the economy. Cluster based on them 
	available_ids = [1,2,4,5,6,7,8,9,10,11,12]
	if get_data == True:
		for i in available_ids:
			get_matrix_data(i,3)
	else:
		pass

	column_name_list = []
	dataframe_list = []
	price_list = []

	for i in available_ids:
		df_coskew = pd.read_sql("SELECT Data,coskew_bench FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(i),acc_engine,index_col = "Data").drop_duplicates()
		df_price = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(i),acc_engine,index_col = "Data").sort_index().drop_duplicates()
		index_name = pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(i),acc_engine).values[0][0]
		column_name_list.append(index_name)
		dataframe_list.append(df_coskew)
		price_list.append(df_price)

	df = pd.concat(dataframe_list, axis=1)
	df.columns = column_name_list
	df.plot(figsize = (15,10))
	plt.title("Coskew sectors with the Market")
	#plt.show()


	df_price_final = pd.concat(price_list,axis =1)
	df_price_final.columns = column_name_list
	df_price_final = df_price_final.resample("M").last()
	df_price_final = df_price_final.pct_change()

	df_last = df.iloc[[-1]].T 
	column_name = df_last.columns[0]
	df_last.rename(columns = {column_name:"last_value"},inplace = True)
	df_sorted = df_last.sort_values(by = ["last_value"])

	## FINISH HERE 
	# JUST PICK THE SORTED VALUE COLUMNS AND DEFINE THE DIFF IN RETURNS TO GET THE SYSTEMATIC COSKEWNEWSS RISK FACTORS	
	high_skew_industries = list(df_sorted.iloc[round(len(df_sorted)/2):len(df_sorted)].index)
	low_skew_industries =list(df_sorted.iloc[0:round(len(df_sorted)/2)].index)

	df_low_skew = df_price_final[low_skew_industries]
	df_low_skew["low_skew"] = df_low_skew.mean(axis = 1)
	df_high_skew = df_price_final[high_skew_industries]
	df_high_skew["high_skew"] = df_high_skew.mean(axis = 1)
	df_skew_factor = df_low_skew["low_skew"]-df_high_skew["high_skew"]
	df_skew_factor = pd.DataFrame(df_skew_factor)
	df_skew_factor.columns =  ["skew_factor"]	
	

	return df_skew_factor

	# 1st select the number for each industry 

#clustering_classification_skew(True)
#asdas

#OUTPUTS the statistics of coskewnewss and correlation over time for the different factors and industry
def industry_funds_coskew(update_data,_type):
	index_ids = list(pd.read_sql("IndexData",acc_engine).sort_values("ID_INDEX")["ID_INDEX"])

	if update_data == True:
		for i in index_ids:
			get_matrix_data(i,3) # 3 is the SP500 total

	else:
		pass
	industry_dict = {"index_value":43,"index_growth":44,"index_small_cap":95,"index_large_cape":96,"index_commodity":34,"index_momentum":97
	,"index_oil_producer":66,"index_metals":64,"index_reit":98,"index_semiconductors":71}
	#59,60,61,62,63,64,65,66,67,68,69,70,71,98

	df_value = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(43),acc_engine,index_col = "Data")
	df_growth = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(44),acc_engine,index_col = "Data")
	df_small = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(95),acc_engine,index_col = "Data")
	df_large = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(96),acc_engine,index_col = "Data")
	df_commodity = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(34),acc_engine,index_col = "Data")
	df_momentum = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(97),acc_engine,index_col = "Data")
	df_oil = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(66),acc_engine,index_col = "Data")
	df_metals = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(64),acc_engine,index_col = "Data")
	df_reit = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(98),acc_engine,index_col = "Data")
	df_semiconductors = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(71),acc_engine,index_col = "Data")

	
	value_name = pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(43),acc_engine).values[0][0]
	growth_name= pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(44),acc_engine).values[0][0]
	small_name= pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(95),acc_engine).values[0][0]
	large_name= pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(96),acc_engine).values[0][0]
	commodity_name= pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(34),acc_engine).values[0][0]
	momentum_name= pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(97),acc_engine).values[0][0]
	oil_name= pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(66),acc_engine).values[0][0]
	metals_name= pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(64),acc_engine).values[0][0]
	reit_name= pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(98),acc_engine).values[0][0]
	semiconductors_name =pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(71),acc_engine).values[0][0]

	df_value.rename(columns = {"coskew_bench":"Coskew_{}".format(value_name),"corr_benchmark":"Correlation_{}".format(value_name)},inplace = True)
	df_growth.rename(columns = {"coskew_bench":"Coskew_{}".format(growth_name),"corr_benchmark":"Correlation_{}".format(growth_name)},inplace = True)
	df_small.rename(columns = {"coskew_bench":"Coskew_{}".format(small_name),"corr_benchmark":"Correlation_{}".format(small_name)},inplace = True)
	df_large.rename(columns = {"coskew_bench":"Coskew_{}".format(large_name),"corr_benchmark":"Correlation_{}".format(large_name)},inplace = True)
	df_commodity.rename(columns = {"coskew_bench":"Coskew_{}".format(commodity_name),"corr_benchmark":"Correlation_{}".format(commodity_name)},inplace = True)
	df_momentum.rename(columns = {"coskew_bench":"Coskew_{}".format(momentum_name),"corr_benchmark":"Correlation_{}".format(momentum_name)},inplace = True)
	df_oil.rename(columns = {"coskew_bench":"Coskew_{}".format(oil_name),"corr_benchmark":"Correlation_{}".format(oil_name)},inplace = True)
	df_metals.rename(columns = {"coskew_bench":"Coskew_{}".format(metals_name),"corr_benchmark":"Correlation_{}".format(metals_name)},inplace = True)
	df_reit.rename(columns = {"coskew_bench":"Coskew_{}".format(reit_name),"corr_benchmark":"Correlation_{}".format(reit_name)},inplace = True)
	df_semiconductors.rename(columns = {"coskew_bench":"Coskew_{}".format(semiconductors_name),"corr_benchmark":"Correlation_{}".format(semiconductors_name)},inplace = True)

	name_values_list_cos = ["Coskew_{}".format(value_name),"Coskew_{}".format(growth_name),
	"Coskew_{}".format(small_name),"Coskew_{}".format(large_name),
	"Coskew_{}".format(commodity_name),"Coskew_{}".format(momentum_name),"Coskew_{}".format(oil_name),"Coskew_{}".format(metals_name),"Coskew_{}".format(reit_name),
	"Coskew_{}".format(semiconductors_name)]
	name_values_list_corr = ["Correlation_{}".format(value_name),"Correlation_{}".format(growth_name),
	"Correlation_{}".format(small_name),"Correlation_{}".format(large_name),
	"Correlation_{}".format(commodity_name),"Correlation_{}".format(momentum_name),"Correlation_{}".format(oil_name),"Correlation_{}".format(metals_name),"Correlation_{}".format(reit_name),
	"Correlation_{}".format(semiconductors_name)]
	

	df_merged_values = df_value.merge(df_growth,how = "inner", right_index = True, left_index = True)
	df_merged_values =df_merged_values.merge(df_small,how = "inner", right_index = True, left_index = True)
	df_merged_values =df_merged_values.merge(df_large,how = "inner", right_index = True, left_index = True)
	df_merged_values =df_merged_values.merge(df_commodity,how = "inner", right_index = True, left_index = True)
	df_merged_values =df_merged_values.merge(df_momentum,how = "inner", right_index = True, left_index = True)
	df_merged_values =df_merged_values.merge(df_oil,how = "inner", right_index = True, left_index = True)
	df_merged_values =df_merged_values.merge(df_metals,how = "inner", right_index = True, left_index = True)
	df_merged_values =df_merged_values.merge(df_reit,how = "inner", right_index = True, left_index = True)
	df_merged_values =df_merged_values.merge(df_semiconductors,how = "inner", right_index = True, left_index = True)


	df_merged_cos = df_merged_values[name_values_list_cos]
	df_merged_corr = df_merged_values[name_values_list_corr]

	df_merged_cos.plot(figsize = (15,10))
	#plt.show()

	df_merged_corr.plot(figsize = (15,10))
	#plt.show()



	## beta estimations and

	df_price_value = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(43),acc_engine,index_col = "Data").sort_index()
	df_price_growth = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(44),acc_engine,index_col = "Data").sort_index()
	df_price_small = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(95),acc_engine,index_col = "Data").sort_index()
	df_price_large = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(96),acc_engine,index_col = "Data").sort_index()
	df_price_commodity = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(34),acc_engine,index_col = "Data").sort_index()
	df_price_momentum = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(97),acc_engine,index_col = "Data").sort_index()
	df_price_oil = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(66),acc_engine,index_col = "Data").sort_index()
	df_price_metals = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(64),acc_engine,index_col = "Data").sort_index()
	df_price_reit = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(98),acc_engine,index_col = "Data").sort_index()
	df_price_semiconductors = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(71),acc_engine,index_col = "Data").sort_index()
	df_price_market =pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(3),acc_engine,index_col = "Data").sort_index()


	df_price_value.rename(columns = {"Close":"Close_{}".format(value_name)},inplace = True)
	df_price_growth.rename(columns = {"Close":"Close_{}".format(growth_name)},inplace = True)
	df_price_small.rename(columns = {"Close":"Close_{}".format(small_name)},inplace = True)
	df_price_large.rename(columns = {"Close":"Close_{}".format(large_name)},inplace = True)
	df_price_commodity.rename(columns = {"Close":"Close_{}".format(commodity_name)},inplace = True)
	df_price_momentum.rename(columns = {"Close":"Close_{}".format(momentum_name)},inplace = True)
	df_price_oil.rename(columns = {"Close":"Close_{}".format(oil_name)},inplace = True)
	df_price_metals.rename(columns = {"Close":"Close_{}".format(metals_name)},inplace = True)
	df_price_reit.rename(columns = {"Close":"Close_{}".format(reit_name)},inplace = True)
	df_price_semiconductors.rename(columns = {"Close":"Close_{}".format(semiconductors_name)},inplace = True)
	df_price_market.rename(columns = {"Close":"Close_market"},inplace = True)

	df_price_merge = df_price_value.merge(df_price_growth,how = "inner", right_index = True, left_index = True)
	df_price_merge =df_price_merge.merge(df_price_small,how = "inner", right_index = True, left_index = True)
	df_price_merge =df_price_merge.merge(df_price_large,how = "inner", right_index = True, left_index = True)
	df_price_merge =df_price_merge.merge(df_price_commodity,how = "inner", right_index = True, left_index = True)
	df_price_merge =df_price_merge.merge(df_price_momentum,how = "inner", right_index = True, left_index = True)
	df_price_merge =df_price_merge.merge(df_price_oil,how = "inner", right_index = True, left_index = True)
	df_price_merge =df_price_merge.merge(df_price_metals,how = "inner", right_index = True, left_index = True)
	df_price_merge =df_price_merge.merge(df_price_reit,how = "inner", right_index = True, left_index = True)
	df_price_merge =df_price_merge.merge(df_price_semiconductors,how = "inner", right_index = True, left_index = True)
	df_price_merge = df_price_merge.merge(df_price_market,how = "inner", right_index = True, left_index = True)

	name_values_list_price = ["Close_{}".format(value_name),"Close_{}".format(growth_name),"Close_{}".format(small_name),
	"Close_{}".format(large_name),"Close_{}".format(commodity_name),
	"Close_{}".format(momentum_name),"Close_{}".format(oil_name),
	"Close_{}".format(metals_name),"Close_{}".format(reit_name),
	"Close_{}".format(semiconductors_name),"Close_market"]

	df_price_merge = df_price_merge[name_values_list_price]

	df_price_merge_m = df_price_merge.resample("M").last()
	df_price_merge_m = df_price_merge.pct_change()
	corr_factors = df_price_merge_m.corr().round(2)
	fig, ax = plt.subplots(figsize=(20,8))
	ax = sns.heatmap(corr_factors, annot=True)
	plt.tight_layout()
	ax.set_title("Correlation Matrix among Factors")
	ax.set_xlabel("Factor ID")
	ax.set_ylabel("Factor ID")
	#plt.show()

	df_price_merge_m["smb"] = df_price_merge_m["Close_{}".format(small_name)]- df_price_merge_m["Close_{}".format(large_name)]
	df_price_merge_m["value_effect"] = df_price_merge_m["Close_{}".format(growth_name)]- df_price_merge_m["Close_{}".format(value_name)]
	df_price_merge_m["momentum_effect"] = df_price_merge_m["Close_{}".format(momentum_name)]
	df_price_merge_m = df_price_merge_m[["smb","value_effect","momentum_effect"]]

	available_ids = [59,60,61,62,63,64,65,66,67,68,69,70,71,98]
	df_market = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(3),acc_engine,index_col = "Data").sort_index()
	df_market = df_market.resample("M").last()
	df_market = df_market.pct_change()
	df_market.rename(columns = {"Close":"Close_market"},inplace = True)
	df_price_merge_m =df_price_merge_m.merge(df_market,how = "inner", right_index = True, left_index = True)

	for i in available_ids:
		df_price = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(i),acc_engine,index_col = "Data").sort_index()
		df_price = df_price.resample("M").last()
		df_price = df_price.pct_change()
		df_coskew_index = pd.read_sql("SELECT Data,unconditional_cos FROM CoSkewnessKurt WHERE ID_INDEX = {}".format(i),acc_engine,index_col = "Data").sort_index()
		df_skew_factor = clustering_classification_skew(False)
		df_price_merge_m = df_price_merge_m.merge(df_price,how = "inner", right_index = True, left_index = True)
		df_price_merge_m = df_price_merge_m.merge(df_coskew_index,how = "inner", right_index = True, left_index = True)
		df_price_merge_m = df_price_merge_m.merge(df_skew_factor,how = "inner", right_index = True, left_index = True)
		df_price_merge_m.dropna(inplace = True)

		index_name = pd.read_sql("SELECT (symbol) FROM IndexData WHERE ID_INDEX = {}".format(i),acc_engine).values[0][0]
		def append_data(_model,variable,variable_name):
			new_frame = df_price_merge_m.reset_index()
			date = new_frame["Data"].iloc[-1]
			list_of_values = [i,_model,date,variable_name,variable]
			df_append = pd.DataFrame([list_of_values],columns = ["ID_INDEX","Model","Data","Variable_name","Variable_value"])
			variable = float(variable)
			df_append.set_index("ID_INDEX",inplace = True)
			date =date.to_pydatetime()
			df_append_exists = pd.read_sql("SELECT * FROM RegressionResults WHERE Variable_value = {} AND Variable_name = '{}' AND Model = '{}' ".format(variable,variable_name,_model),acc_engine)
			if df_append_exists.empty:
				df_append.to_sql("RegressionResults",acc_engine,if_exists = "append")
			else:
				pass

		if _type == "Coskew_effect":
			X = np.c_[df_price_merge_m[["Close_market","skew_factor"]]]
			Y = np.c_[df_price_merge_m["Close"]]
			polynomial_features = PolynomialFeatures(degree=1)
			X = polynomial_features.fit_transform(X)

			lin_reg_model = sklearn.linear_model.LinearRegression().fit(X,Y)
			beta_1 = lin_reg_model.coef_[0][1]
			beta_2 = lin_reg_model.coef_[0][2]

			LR = LinearRegression()
			LR.fit(X,Y)
			y_prediction = LR.predict(X)
			r2 = r2_score(Y,y_prediction)
			est = sm.OLS(Y, X)
			est = est.fit()
			p_values = est.summary2().tables[1]['P>|t|']

			total_beta = beta_1/r2
			r = np.zeros_like(est.params)
			T_test = est.t_test(r)
			df_price_merge_m.drop(columns = "unconditional_cos",inplace = True)
			print("The regression results for index ({})".format(index_name))
			print(est.summary())
			p_value_market = p_values.iloc[1]
			p_value_coskew = p_values.iloc[-1]
			df_price_merge_m.drop(columns ="Close",inplace = True)
			df_price_merge_m.drop(columns ="skew_factor",inplace = True)

			append_data(_type,r2,"R2")
			append_data(_type,beta_1,"beta_market")
			append_data(_type,beta_2,"beta_skew")
			append_data(_type,p_value_market,"p_value beta_market")
			append_data(_type,p_value_coskew,"p_value beta_skew")


		elif _type == "4 Factor Model":
			X = np.c_[df_price_merge_m[["Close_market","smb","value_effect","skew_factor"]]]
			Y = np.c_[df_price_merge_m["Close"]]
			polynomial_features = PolynomialFeatures(degree=1)
			X = polynomial_features.fit_transform(X)

			lin_reg_model = sklearn.linear_model.LinearRegression().fit(X,Y)
			beta_1 = lin_reg_model.coef_[0][1]
			beta_2 = lin_reg_model.coef_[0][2]
			beta_3 = lin_reg_model.coef_[0][3]
			beta_4 = lin_reg_model.coef_[0][4]

			LR = LinearRegression()
			LR.fit(X,Y)
			y_prediction = LR.predict(X)
			r2 = r2_score(Y,y_prediction)
			est = sm.OLS(Y, X)
			est = est.fit()
			total_beta = beta_1/r2
			r = np.zeros_like(est.params)
			p_values = est.summary2().tables[1]['P>|t|']
			p_value_market = p_values.iloc[1]
			p_value_smb = p_values.iloc[2]
			p_value_value = p_values.iloc[3]
			p_value_coskew = p_values.iloc[-1]

			df_price_merge_m.drop(columns = "unconditional_cos",inplace = True)
			print("The regression results for index ({})".format(index_name))
			print(est.summary())
			df_price_merge_m.drop(columns ="Close",inplace = True)
			df_price_merge_m.drop(columns ="skew_factor",inplace = True)
			append_data(_type,r2,"R2")
			append_data(_type,beta_1,"beta_market")
			append_data(_type,beta_2,"beta_smb")
			append_data(_type,beta_3,"beta_value")
			append_data(_type,beta_4,"beta_skew_ff")
			append_data(_type,p_value_market,"p_value beta_market")
			append_data(_type,p_value_smb,"p_value beta_smb")
			append_data(_type,p_value_value,"p_value beta_value")
			append_data(_type,p_value_coskew,"p_value beta_coskew")

		
		else:
			pass



def fund_timing_ability(id_instrument):
	df_fund = pd.read_sql("SELECT * FROM FundPriceData WHERE ID_FUND = {}".format(id_instrument), acc_engine, index_col = "Data").sort_index()
	df_market = pd.read_sql("SELECT Data,Close FROM IndexPriceData WHERE ID_INDEX = {}".format(3),acc_engine,index_col = "Data").sort_index()




def main():
	# True updates the data. False just calculates the stats
	#industry_funds_coskew(False,"Coskew_effect")
	#industry_funds_coskew(False,"4 Factor Model")
	clustering_classification_skew(False)



