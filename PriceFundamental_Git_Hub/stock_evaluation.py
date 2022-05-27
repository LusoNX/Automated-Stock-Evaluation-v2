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
from scipy.stats import norm,skewnorm
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import operator
import statsmodels.formula.api as smf
import fund_evaluation
import evaluation_tools

conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=YOUR DIRECTORY\StockPriceFundamentalData.accdb;')
cnn_url = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
acc_engine = create_engine(cnn_url)


conn_str_funds = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=YOUR DIRECTORY\FundDataBase.accdb;')
cnn_url_funds = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str_funds)}"
acc_engine_funds = create_engine(cnn_url_funds)


def get_matrix_data(id_bench):
	matrix_directory = r"YOUR DIRECTORY\Matrix_figures"
	stock_ids = list(pd.read_sql("SELECT (ID_STOCK) FROM StockIndex ",acc_engine).sort_values("ID_STOCK")["ID_STOCK"])
	today_date = date.today()
	beg_date =  date.today() - relativedelta(years=3)
	today_date = np.datetime64(today_date)
	beg_date = np.datetime64(beg_date)
	df_benchmark = pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH ={}".format(id_bench),acc_engine,index_col = "Data").sort_index()
	df_benchmark.rename(columns = {"Close":"Close_market"},inplace = True)
	df_benchmark = df_benchmark[["Close_market"]]


	date_range = pd.date_range(beg_date, today_date, freq='D')
	df_returns_all = pd.DataFrame(index = date_range)

	for i in stock_ids:
		df_price = pd.read_sql("SELECT Data,Close FROM PriceData WHERE ID_STOCK = {} ".format(i),acc_engine).sort_values("Data")
		df_price.set_index("Data",inplace = True)
		df_price_y = df_price.resample("Y").last()
		nr_years = len(df_price_y)
		id_name = str(i)
		df_price[id_name] = df_price["Close"].pct_change()
		df_price.dropna(inplace = True)
		if nr_years >= 3 :
			df_new = df_price[[id_name]]
			df_returns_all = df_new.merge(df_returns_all,how = "inner", right_index = True, left_index = True)
		else:
			pass
	

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
	plt.show()
	plt.savefig(matrix_directory+r"\kurt_matrix_{}.png".format(today_date))
	plt.close()


	## Get Cosk for market data
	for x in stock_ids:
		df_price = pd.read_sql("SELECT Data,Close FROM PriceData WHERE ID_STOCK = {} ".format(x),acc_engine,index_col = "Data").sort_values("Data")
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


		df_coskew_stock["ID_STOCK"] =x
		df_coskew_stock["ID_BENCH"] =id_bench
		df_coskew_stock.reset_index(inplace = True)
		df_coskew_stock.set_index("ID_STOCK",inplace = True)
		df_coskew_stock = df_coskew_stock.sort_values("Data")
		df_coskew_stock_exists = pd.read_sql("SELECT * FROM CoSkewnessKurt WHERE ID_STOCK= {} AND ID_BENCH = {}".format(x,id_bench),acc_engine).sort_values("Data")
		if df_coskew_stock_exists.empty:
			df_coskew_stock.to_sql("CoSkewnessKurt",acc_engine,if_exists = "append")
		else:
			last_date = df_coskew_stock_exists["Data"].iloc[-1]
			last_date =np.datetime64(last_date)
			mask_2  = (df_coskew_stock["Data"] > last_date)
			df_coskew_stock = df_coskew_stock[mask_2]
			df_coskew_stock.to_sql("CoSkewnessKurt",acc_engine,if_exists = "append")




def get_betas(id_instrument,_type,window_frame,beta_years):
	df_price = pd.read_sql("SELECT * FROM PriceData WHERE ID_STOCK = {}".format(id_instrument),acc_engine,index_col = "Data").sort_values(by = "Data")
	df_market= pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH = {}".format(4),acc_engine,index_col = "Data").sort_values(by = "Data") #4 is the number for the market returns
	id_industry = pd.read_sql("SELECT (ID_INDUSTRY) FROM IndustryStockMerge WHERE ID_STOCK = {}".format(id_instrument),acc_engine).values[0][0]
	df_industry = pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH = {}".format(id_industry),acc_engine,index_col = "Data").sort_values(by = "Data")
	df_industry.rename(columns = {"Close":"Close_industry"},inplace = True)
	df_small = pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH = {}".format(100),acc_engine,index_col = "Data").sort_values(by = "Data")
	df_small.rename(columns = {"Close":"Close_small"},inplace = True)
	df_big = pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH = {}".format(99),acc_engine,index_col = "Data").sort_values(by = "Data")
	df_big.rename(columns = {"Close":"Close_big"},inplace = True)
	df_value = pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH = {}".format(101),acc_engine,index_col = "Data").sort_values(by = "Data")
	df_value.rename(columns = {"Close":"Close_value"},inplace = True)
	df_growth =pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH = {}".format(102),acc_engine,index_col = "Data").sort_values(by = "Data")
	df_growth.rename(columns = {"Close":"Close_growth"},inplace = True)
	df_volatility =pd.read_sql("SELECT * FROM IndexPriceData WHERE ID_INDEX = {}".format(13),acc_engine_funds,index_col = "Data").sort_values(by = "Data")
	df_volatility.rename(columns = {"Close":"Close_volatility"},inplace = True)
	df_momentum = pd.read_sql("SELECT Data, Close FROM FactorPriceData WHERE ID_FACTOR ={}".format(2),acc_engine_funds,index_col = "Data").sort_index().resample("W").last()
	df_momentum.rename(columns = {"Close":"Close_momentum"},inplace = True)
	df_values = df_price.merge(df_market,how = "inner", right_index = True, left_index = True)
	df_values = df_values.merge(df_industry, how = "inner",right_index = True, left_index = True)
	df_values = df_values.merge(df_small, how = "inner",right_index = True, left_index = True)
	df_values = df_values.merge(df_big, how = "inner",right_index = True, left_index = True)
	df_values = df_values.merge(df_value, how = "inner",right_index = True, left_index = True)
	df_values = df_values.merge(df_growth, how = "inner",right_index = True, left_index = True)

	df_values_m = df_values.resample("W").last()
	df_values_m["returns_stock"] = df_values_m["Close_x"].pct_change()
	df_values_m["returns_market"] =df_values_m["Close_y"].pct_change()
	df_values_m["returns_industry"] = df_values_m["Close_industry"].pct_change()
	df_values_m["returns_small"] = df_values_m["Close_small"].pct_change()
	df_values_m["returns_big"] = df_values_m["Close_big"].pct_change()
	df_values_m["returns_value"] = df_values_m["Close_value"].pct_change()
	df_values_m["returns_growth"] = df_values_m["Close_growth"].pct_change()
	df_values_m["smb"] = df_values_m["returns_small"] - df_values_m["returns_big"]
	df_values_m["value_effect"] = df_values_m["returns_growth"]-df_values_m["returns_value"]


	df_values_m = df_values_m[["returns_stock","returns_market","returns_industry","smb","value_effect"]]
	df_values_larger = df_values_m
	df_values_larger = df_values_larger.iloc[1::]
	df_momentum.rename(columns = {"Close":"momentum_factor"},inplace = True)
	df_momentum = df_momentum.pct_change()
	df_values_m = df_values_m.merge(df_momentum, how = "inner",right_index = True, left_index = True)
	df_values_m = df_values_m.iloc[1::]
	df_values_m.dropna(inplace = True)


	
	# Momentum Factor
	beta_1_list = []
	beta_2_list = []
	beta_3_list = []
	p_value_1_list = []
	p_value_2_list = []
	p_value_3_list = []
	total_beta_list = []
	r2_list = []
	alpha_list = []
	if _type == "Market":
		range_periods = len(df_values_larger)-window_frame
		df_values_index = df_values_larger.iloc[-range_periods:]
		index_reg = df_values_index.index

		beta_frame = window_frame*beta_years
		for i in range(0,range_periods):
			df_values_n = df_values_larger.iloc[0+i:beta_frame+i]
			X = df_values_n["returns_market"]
			Y = df_values_n["returns_stock"]
			polynomial_features = PolynomialFeatures(degree=1)
			model = sm.OLS(Y, X)
			results = model.fit()
			parameters = results.params.T
			r2 = results.rsquared
			beta_market = parameters["returns_market"]
			r2_list.append(r2)
			beta_1_list.append(beta_market)
			total_beta = beta_market/r2
			total_beta_list.append(total_beta)
			
		df_regression = pd.DataFrame({"Beta_Market":beta_1_list,"R2":r2_list, "total_beta":total_beta_list},index = index_reg)
		return df_regression,df_values_larger
		

	elif _type =="Market 3 Moment":
		beta_frame = window_frame*beta_years
		range_periods = len(df_values_larger)-beta_frame
		df_values_index = df_values_larger.iloc[-range_periods:]
		index_reg = df_values_index.index
		beta_cosk_list = []
		p_values_2 = []
		for i in range(0,range_periods):
			df_values_n = df_values_larger.iloc[0+i:beta_frame+i]
			X = df_values_n[["returns_market"]]
			Y = df_values_n["returns_stock"]
			poly = PolynomialFeatures(2)
			X = poly.fit_transform(X)
			model = sm.OLS(Y, X)
			results = model.fit()
			parameters = results.params.T
			r2 = results.rsquared
			p_values = results.pvalues
			beta_market = parameters["x1"]

			beta_cosk = parameters["x2"]
			p_values_2.append(p_values["x2"])
			beta_cosk_list.append(beta_cosk)
			r2_list.append(r2)
			beta_1_list.append(beta_market)
			total_beta = beta_market/r2
			total_beta_list.append(total_beta)
		
		df_regression_results = pd.DataFrame({"Beta_Market":beta_1_list,"Beta_Cosk":beta_cosk_list,
								"p_values_cosk":p_values_2,"R2":r2_list,"total_beta":total_beta_list}
								,index = index_reg)

		return df_regression_results,df_values_larger


	elif _type == "Carhart":
		beta_frame = window_frame*beta_years
		range_periods = len(df_values_m)-beta_frame
		df_values_index = df_values_m.iloc[-range_periods:]
		index_reg = df_values_index.index
		p_values_2 = []
		p_values_3 = []
		p_values_4 = []
		beta_4_list = []
		df_values_m.rename(columns = {"Close_momentum":"momentum_factor"},inplace = True)
		for i in range(0,range_periods):
			df_values_n = df_values_m.iloc[0+i:beta_frame+i]
			X = df_values_n[["returns_market","smb","value_effect","momentum_factor"]]
			Y = df_values_n["returns_stock"]
			polynomial_features = PolynomialFeatures(degree=1)
			X = polynomial_features.fit_transform(X)
			model = sm.OLS(Y, X)
			results = model.fit()
			parameters = results.params.T
			r2 = results.rsquared
			p_values = results.pvalues
			beta_market = parameters["x1"]
			beta_smb = parameters["x2"]
			beta_hml = parameters["x3"]
			beta_momentum = parameters["x4"]
			p_values_2.append(p_values["x2"])
			p_values_3.append(p_values["x3"])
			p_values_4.append(p_values["x4"])
			beta_1_list.append(beta_market)
			beta_2_list.append(beta_smb)
			beta_3_list.append(beta_hml)
			beta_4_list.append(beta_momentum)
			r2_list.append(r2)

		df_regression = pd.DataFrame({"Beta_Market":beta_1_list,"Beta_smb":beta_2_list,
							"Beta_hml":beta_3_list,"Beta_Momentum":beta_4_list,"R2":r2_list,"p_values_smb":p_values_2,"p_values_hml":p_values_3,"p_values_momentum":p_values_4}
							,index = index_reg)

	
		
		return df_regression,df_values_m
	

	elif _type == "Industry Fama and French":
		beta_frame = window_frame*beta_years
		range_periods = len(df_values_m)-beta_frame
		df_values_index = df_values_m.iloc[-range_periods:]
		index_reg = df_values_index.index
		p_values_2 = []
		p_values_3 = []
		p_values_1 = []
		for i in range(0,range_periods):
			df_values_n = df_values_larger.iloc[0+i:beta_frame+i]
			X = np.c_[df_values_n[["returns_industry","smb","value_effect"]]]
			Y = df_values_n["returns_stock"]
			polynomial_features = PolynomialFeatures(degree=1)
			X = polynomial_features.fit_transform(X)
			model = sm.OLS(Y, X)
			results = model.fit()
			parameters = results.params.T
			r2 = results.rsquared
			p_values = results.pvalues
			beta_industry = parameters["x1"]
			beta_smb = parameters["x2"]
			beta_hml = parameters["x3"]
			p_values_2.append(p_values["x2"])
			p_values_3.append(p_values["x3"])
			p_values_1.append(p_values["x1"])
			beta_1_list.append(beta_industry)
			beta_2_list.append(beta_smb)
			beta_3_list.append(beta_hml)
			r2_list.append(r2)

		df_regression = pd.DataFrame({"Beta_Industry":beta_1_list,"Beta_smb":beta_2_list,
							"Beta_hml":beta_3_list,"R2":r2_list,"p_values_smb":p_values_2,"p_values_hml":p_values_3,"p_values_industry":p_values_1}
							,index = index_reg)

	
		
		return df_regression,df_values_larger


	
def get_beta_figures(id_stock,_type,rolling_window):
	symbol = pd.read_sql("SELECT (SYMBOL) FROM StockIndex WHERE ID_STOCK= {}".format(id_stock),acc_engine)["SYMBOL"].values[0]
	number_of_years = int(rolling_window/52)

	if _type == "Market 3 Moment":
		df_regression,df_returns = get_betas(id_stock,"Market 3 Moment",rolling_window,1) # Change 3 to id_stock
		mean_beta = df_regression["Beta_Market"].mean()
		std_beta = df_regression["Beta_Market"].std()
		scaler = StandardScaler()
		df_regression["Beta_Cosk_Standardized"] = scaler.fit_transform(df_regression[["Beta_Cosk"]])
		df_regression[["Beta_Market","Beta_Cosk_Standardized"]].plot()
		directory = r"C:\Users\35196\OneDrive\Desktop\Investment Prospectus\DataBase\Stock\PriceFundamental\{}".format(symbol)
		plt.legend(loc = "best")
		plt.title("Beta 3 (Semanal) Moment Model (Rolling Window {} Anos)".format(number_of_years))
		plt.xlabel("Data")
		plt.ylabel("Valores")
		plt.savefig(directory +r"\beta_3_moment_{}".format(symbol))
		plt.close("all")


		last_beta = df_regression["Beta_Market"].iloc[-1]
		last_cosk = df_regression["Beta_Cosk"].iloc[-1]
		r2 = df_regression["R2"].iloc[-1]
		beta_total = df_regression["total_beta"].iloc[-1]

		df_regression["p_values_cosk"].plot()
		plt.title("P-Values fator Coskewnewss (Rm**2)")
		plt.xlabel("Data")
		plt.ylabel("Valores")
		plt.savefig(directory +r"\p_values_3_moment_{}".format(symbol))
		plt.close("all")

		p_values_cosk = df_regression["p_values_cosk"].iloc[-1]
		list_of_values = [mean_beta,std_beta,last_beta,last_cosk,beta_total,p_values_cosk,r2]
		list_of_values = [str(round(num, 2)) for num in list_of_values]
		return list_of_values

	elif _type == "Carhart":
		df_regression_4_factor,df_returns = get_betas(id_stock,"Carhart",rolling_window,1)
		df_regression_4_factor[["Beta_Market","Beta_smb","Beta_hml","Beta_Momentum"]].plot()
		directory = r"C:\Users\35196\OneDrive\Desktop\Investment Prospectus\DataBase\Stock\PriceFundamental\{}".format(symbol)
		plt.legend(loc = "best")
		plt.title("Beta 4 (Semanal) Factor Carhart (Rolling Window {} Anos)".format(number_of_years))
		plt.xlabel("Data")
		plt.ylabel("Valores")
		plt.savefig(directory +r"\beta_4_moment_{}".format(symbol))
		plt.close("all")

		number_of_years = int(rolling_window/52)
		df_regression_4_factor["p_values_smb"].plot()
		df_regression_4_factor["p_values_hml"].plot()
		df_regression_4_factor["p_values_momentum"].plot()
		plt.legend(loc = "best")
		plt.title("p_values (Rolling Window {} Anos)".format(number_of_years))
		plt.xlabel("Data")
		plt.ylabel("Valores")
		plt.savefig(directory +r"\pvalues_4_moment_{}".format(symbol))
		plt.close("all")

		last_beta_market = df_regression_4_factor["Beta_Market"].iloc[-1]
		last_beta_smb= df_regression_4_factor["Beta_smb"].iloc[-1]
		last_beta_hml= df_regression_4_factor["Beta_hml"].iloc[-1]
		last_beta_momentum = df_regression_4_factor["Beta_Momentum"].iloc[-1]
		r2 = df_regression_4_factor["R2"].iloc[-1]

		p_values_smb = df_regression_4_factor["p_values_smb"].iloc[-1]
		p_values_hml =df_regression_4_factor["p_values_hml"].iloc[-1]
		p_values_momentum =df_regression_4_factor["p_values_momentum"].iloc[-1]

		list_of_values = [last_beta_market,last_beta_smb,last_beta_hml,last_beta_momentum,r2,p_values_smb,p_values_hml,p_values_momentum]
		list_of_values = [str(round(num, 2)) for num in list_of_values]
		return list_of_values


	elif _type == "Industry Fama and French":
		df_regression,df_returns = get_betas(id_stock,"Industry Fama and French",rolling_window,1)
		directory = r"C:\Users\35196\OneDrive\Desktop\Investment Prospectus\DataBase\Stock\PriceFundamental\{}".format(symbol)
		df_regression[["Beta_Industry","Beta_smb","Beta_hml"]].plot()
		plt.legend(loc = "best")
		plt.title("Beta Industry Fama and French (Rolling Window {} Anos)".format(number_of_years))
		plt.xlabel("Data")
		plt.ylabel("Valores")
		plt.savefig(directory +r"\beta_industry_ff_{}".format(symbol))
		plt.close("all")

		number_of_years = int(rolling_window/52)
		df_regression["p_values_industry"].plot()
		df_regression["p_values_smb"].plot()
		df_regression["p_values_hml"].plot()
		plt.legend(loc = "best")
		plt.title("p_values (Rolling Window {} Anos)".format(number_of_years))
		plt.xlabel("Data")
		plt.ylabel("Valores")
		plt.savefig(directory +r"\pvalues_industry_ff_{}".format(symbol))
		plt.close("all")

		last_beta_industry = df_regression["Beta_Industry"].iloc[-1]
		last_beta_smb= df_regression["Beta_smb"].iloc[-1]
		last_beta_hml= df_regression["Beta_hml"].iloc[-1]
		r2 = df_regression["R2"].iloc[-1]

		p_values_smb = df_regression["p_values_smb"].iloc[-1]
		p_values_hml =df_regression["p_values_hml"].iloc[-1]
		p_values_industry = df_regression["p_values_industry"].iloc[-1]

		list_of_values = [last_beta_industry,last_beta_smb,last_beta_hml,r2,p_values_smb,p_values_hml,p_values_industry]
		list_of_values = [str(round(num, 2)) for num in list_of_values]
		return list_of_values

	else:
		pass






def company_cash_flows(id_symbol,_tax_rate):
	df_balance = pd.read_sql("SELECT * FROM BalanceSheetData WHERE ID_STOCK = {} AND period_type = '{}' ".format(id_symbol,"Annual"),acc_engine,index_col = "ID_STOCK").sort_values(by="Data")
	df_financials = pd.read_sql("SELECT * FROM IncomeStatementData WHERE ID_STOCK = {} AND period_type = '{}' ".format(id_symbol,"Annual"),acc_engine,index_col = "ID_STOCK").sort_values(by="Data")
	df_cash_flow = pd.read_sql("SELECT * FROM CashFlowData WHERE ID_STOCK = {} AND period_type = '{}' ".format(id_symbol,"Annual"),acc_engine,index_col = "ID_STOCK").sort_values(by="Data")
	df_balance["operating_assets"] = df_balance["totalCurrentAssets"] - df_balance["cashAndShortTermInvestments"] 
	df_balance["fixed_assets"] = df_balance["propertyPlantEquipment"]
	df_balance["intangibles"] = df_balance["intangibleAssets"]
	df_balance["cash_and_equiv"] = df_balance["cashAndShortTermInvestments"]
	df_balance["LT_investments"] = df_balance["longTermInvestments"]
	df_balance["other_assets"] = df_balance["otherNonCurrrentAssets"]
	df_balance["non_operating_assets"] = df_balance["other_assets"]+df_balance["LT_investments"]+df_balance["cash_and_equiv"]+df_balance["intangibles"]
	df_balance["operating_liabilities"] = df_balance["totalCurrentLiabilities"] - df_balance["shortTermDebt"] + df_balance["otherNonCurrentLiabilities"]
	df_balance["debt"] = df_balance["capitalLeaseObligations"] +df_balance["shortLongTermDebtTotal"]
	

	# Checks if sum of the weights are close to 1
	df_assets = df_balance[["Data","cashAndShortTermInvestments","operating_assets","fixed_assets","intangibles","cash_and_equiv","LT_investments","other_assets","non_operating_assets","operating_liabilities","debt","totalShareholderEquity","totalAssets","commonStockSharesOutstanding"]]
	df_assets["operating_asssets_perc"] = df_assets["operating_assets"] / df_assets["totalAssets"]
	df_assets["fixed_assets_perc"] = df_assets["fixed_assets"] / df_assets["totalAssets"]
	df_assets["intangibles_perc"] = df_assets["intangibles"] / df_assets["totalAssets"]
	df_assets["cash_and_equiv_perc"] = df_assets["cashAndShortTermInvestments"] / df_assets["totalAssets"]
	df_assets["LT_investments_perc"] = df_assets["LT_investments"] / df_assets["totalAssets"]
	df_assets["other_assets_perc"] = df_assets["other_assets"] / df_assets["totalAssets"]
	df_assets["non_operating_assets_perc"] = df_assets["non_operating_assets"] / df_assets["totalAssets"]
	df_assets["debt_perc"] = df_assets["debt"] / df_assets["totalAssets"]
	df_assets["operating_liabilities_perc"] = df_assets["operating_liabilities"] / df_assets["totalAssets"]
	df_assets["equity_perc"] = df_assets["totalShareholderEquity"] / df_assets["totalAssets"]

	## IDEIA TO ADD - FILTER OUT ONLY STOCKS IN WHICH THE SUM OF THE WEIGHTS IS CLOSE TO 1. 

	df_assets["working_capital"] = df_assets["operating_assets"] - df_assets["operating_liabilities"]
	df_assets["nwc"] = df_assets["working_capital"].diff()
	df_assets["capex"] = df_assets["fixed_assets"].diff()
	df_assets["new_debt"] = df_assets["debt"].diff()

	df_financials.fillna(0,inplace = True)
	df_final = pd.DataFrame(index = df_assets.index)
	df_final["Data"] = df_assets["Data"]
	df_final["ebit_after_tax"] = df_financials["ebit"]*(1-_tax_rate)
	df_final["fcff"] = df_final["ebit_after_tax"] - df_assets["nwc"] - df_assets["capex"]
	df_final["fcfe"] = df_final["fcff"] + df_assets["new_debt"] - (df_financials["interestAndDebtExpense"]*(1-_tax_rate))
	df_final["fcff_per_share"] = df_final["fcff"] / df_assets["commonStockSharesOutstanding"]
	df_final["fcfe_per_share"] = df_final["fcfe"] / df_assets["commonStockSharesOutstanding"]
	df_final["cash_per_share"] = df_assets["cash_and_equiv"] / df_assets["commonStockSharesOutstanding"]

	cash_per_share = df_final["cash_per_share"].iloc[-1]
	df_ratio = pd.read_sql("SELECT * FROM RatioData WHERE ID_STOCK = {}".format(id_symbol),acc_engine)
	shares_out = df_assets["commonStockSharesOutstanding"].iloc[-1]
	fcf = df_ratio["fcf"].iloc[-1]

	return df_final,fcf,shares_out,cash_per_share

def CAGR_ret(DF,columns_ret,_period):
	df = DF.copy()
	mean_ret =df[columns_ret].mean()
	cagr = ((1+mean_ret)**(_period))-1
	return cagr





def cash_flow_valuation_model(id_instrument,beta_model,valuation_model,iterations):
	symbol = pd.read_sql("SELECT Symbol FROM StockIndex WHERE ID_STOCK = {}".format(id_instrument),acc_engine)["Symbol"].values[0]
	directory = r"C:\Users\35196\OneDrive\Desktop\Investment Prospectus\DataBase\Stock\PriceFundamental\{}".format(symbol)
	df_market = pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH = {} ".format(4),acc_engine,index_col = "Data").sort_values("Data")
	df_market = df_market.resample("W").last()
	df_market = df_market.iloc[-1040:]
	best_fit_market = evaluation_tools.best_fit(df_market,"Close","returns")
	df_stock_price = pd.read_sql("SELECT * FROM PriceData WHERE ID_STOCK = {} ".format(id_instrument),acc_engine,index_col = "Data").sort_values("Data")
	last_price = df_stock_price["Close"].iloc[-1]

	plt.close("all")

	def get_div_g_rate(id_instrument):
		df_ratio = pd.read_sql("SELECT * FROM RatioData WHERE ID_STOCK = {}".format(id_instrument),acc_engine)
		div_rate = df_ratio["div_rate"].iloc[-1]
		div = last_price*div_rate
		df_div_quarter = pd.read_sql("SELECT Data,dividendPayout,period_type FROM CashFlowData WHERE ID_STOCK = {} AND period_type ='{}'".format(id_instrument,"Quarter"),acc_engine).sort_values("Data")
		df_div_quarter["dividendPayout"]  =df_div_quarter["dividendPayout"].astype(int)		
		cagr_div = (df_div_quarter["dividendPayout"].iloc[-1] /df_div_quarter["dividendPayout"].iloc[0]) *(1/(4*len(df_div_quarter)))
		std_div_g = df_div_quarter["dividendPayout"].std()*2
		left_g = cagr_div - (cagr_div/2)
		right_g = cagr_div + (cagr_div/2)
		growth_rate_dist = np.random.triangular(left_g,cagr_div,right_g, iterations)
		return growth_rate_dist,div

	def get_wacc(ke):
		df_balance = pd.read_sql("SELECT * FROM BalanceSheetData WHERE ID_STOCK = {} AND period_type = '{}' ".format(id_instrument,"Annual"),acc_engine,index_col = "ID_STOCK").sort_values(by="Data")
		df_financials = pd.read_sql("SELECT * FROM IncomeStatementData WHERE ID_STOCK = {} AND period_type = '{}' ".format(id_instrument,"Annual"),acc_engine,index_col = "ID_STOCK").sort_values(by="Data")
		interest_exp = df_financials["interestAndDebtExpense"].iloc[-1]
		total_equity = df_balance["totalShareholderEquity"].iloc[-1]
		total_debt = df_balance["shortLongTermDebtTotal"].iloc[-1]
		perc_debt = total_debt/(total_equity+total_debt)
		perc_eq = total_equity/(total_equity+total_debt)
		kd =interest_exp/total_debt
		wacc = perc_eq*(ke)+perc_debt*kd*(1-0.23)
		return wacc,total_debt


	if beta_model == "Carhart":
		df_regression,df_returns = get_betas(id_instrument,"Carhart",52,1)
		df_factors = df_returns.resample("W").last()
		df_factors = df_factors.sort_index()
		cagr_market = CAGR_ret(df_factors,"returns_market",52)
		cagr_smb= CAGR_ret(df_factors,"value_effect",52)
		cagr_hml = CAGR_ret(df_factors,"smb",52)
		cagr_momentum= CAGR_ret(df_factors,"momentum_factor",52)
		beta_market = df_regression["Beta_Market"].iloc[-1]
		beta_smb =df_regression["Beta_smb"].iloc[-1]
		beta_hml =df_regression["Beta_hml"].iloc[-1]
		beta_momentum = df_regression["Beta_Momentum"].iloc[-1]
		ke = beta_market*cagr_market +cagr_smb*beta_smb + cagr_hml*beta_hml + cagr_momentum*beta_momentum

		if valuation_model == "DDM":
			growth_rate_dist,div = get_div_g_rate(id_instrument)
			if div == 0:
				mean_value = 0 

			else:
				output_distributions = []
				for i in range(iterations):
					new_g = growth_rate_dist[i]		
					discount_factor = ke - new_g
					value_per_share = div*(1+new_g)/discount_factor
					value_per_share = value_per_share.values[0]
					output_distributions.append(value_per_share)
				
				perc_25 = np.percentile(output_distributions,25)
				mean_value = np.mean(output_distributions)
				plt.hist(output_distributions, bins = 100, density = False)
				plt.axvline(last_price, color='k', linestyle='dashed', linewidth=1)
				plt.xlim([0,40])
				plt.savefig(directory+r"\valuation_ddm_carhart_{}".format(symbol))

			return mean_value



		elif valuation_model == "FCFE":
			df,fcf,shares_out,cash_per_share =company_cash_flows(3,0.23)
			fcfe = df["fcfe_per_share"].iloc[-1]
			value_per_share = fcfe*(1+0.02)/(ke-0.02)
			return value_per_share

		elif valuation_model == "FCF":
			df,fcf,shares_out,cash_per_share =company_cash_flows(3,0.23)
			fcf_per_share = fcf/shares_out
			wacc,debt = get_wacc(ke)
			debt_per_share = debt/shares_out
			value_per_share = (fcf_per_share*(1+0.02)/(wacc-0.02)) - debt_per_share +cash_per_share
			return value_per_share

		else:
			pass 





	elif beta_model =="Market 3 Moment":
		df_regression,df_returns = get_betas(id_instrument,"Market 3 Moment",52,1)
		df_merge = df_regression[["Beta_Market","Beta_Cosk"]]
		df_market["returns_market"] = df_market["Close"].pct_change(52)

		df_market["returns_market_2"] =df_market["returns_market"]**2
		df_market.dropna(inplace = True)
		## Check best distribution
		mean_ret_mkt = df_market["returns_market"].mean()
		std_ret_mkt = df_market["returns_market"].std()
		skew_market =df_market["returns_market"].skew()
		last_cosk = df_market["returns_market_2"].iloc[-1]

		def skew_norm_pdf(x,e,w,a):
		# adapated from:
		# http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
			t = (x-e) / w
			return 2.0 * w * stats.norm.pdf(t) * stats.norm.cdf(a*t)


		df_annual_data = df_market.Close.pct_change(52)
		skew_market = df_annual_data.skew()
		#coskew_dist = skew_norm_pdf(df_market["returns_market"],mean_ret_mkt,std_ret_mkt,skew_market)
		plt.close("all")

		beta_market = df_regression["Beta_Market"].mean()
		total_beta = df_regression["total_beta"].iloc[-1]
		beta_cosk = df_regression["Beta_Cosk"].mean()
		last_coskew_market = df_market["returns_market_2"].mean()
		ke = beta_market*mean_ret_mkt+  beta_cosk*last_coskew_market
		output_distributions = []

		if valuation_model == "DDM":
			growth_rate_dist,div = get_div_g_rate(id_instrument)
			

			if div == 0:
				mean_value = 0
				pass 
			else:

				for i in range(iterations):
					new_g = growth_rate_dist[i]		
					discount_factor = ke - new_g
					value_per_share = div*(1+new_g)/discount_factor
					value_per_share = value_per_share.values[0]
					output_distributions.append(value_per_share)
				
				perc_25 = np.percentile(output_distributions,25)
				mean_value = np.mean(output_distributions)
				plt.hist(output_distributions, bins = 100, density = False)
				plt.axvline(last_price, color='k', linestyle='dashed', linewidth=1)
				plt.xlim([25,40])
				plt.savefig(directory+r"\valuation_ddm_3_moment_{}".format(symbol))
			return mean_value


		elif valuation_model == "FCFE":
			df,fcf,shares_out,cash_per_share =company_cash_flows(3,0.23)
			fcfe = df["fcfe_per_share"].iloc[-1]
			value_per_share = fcfe*(1+0.02)/(ke-0.02)
			return value_per_share

		elif valuation_model == "FCF":
			df,fcf,shares_out,cash_per_share =company_cash_flows(3,0.23)
			fcf_per_share = fcf/shares_out
			wacc,debt = get_wacc(ke)
			debt_per_share = debt/shares_out
			value_per_share = (fcf_per_share*(1+0.02)/(wacc-0.02)) - debt_per_share +cash_per_share
			return value_per_share

		else:
			pass 


	elif beta_model == "Industry Fama and French":
		df_regression,df_returns = get_betas(id_instrument,beta_model,52,1)
		df_factors = df_returns[["returns_industry","smb","value_effect"]]
		cagr_industry = CAGR_ret(df_factors,"returns_industry",52)
		cagr_smb= CAGR_ret(df_factors,"value_effect",52)
		cagr_hml = CAGR_ret(df_factors,"smb",52)
		beta_industry = df_regression["Beta_Industry"].iloc[-1]
		beta_smb =df_regression["Beta_smb"].iloc[-1]
		beta_hml =df_regression["Beta_hml"].iloc[-1]
		ke = beta_industry*cagr_industry +cagr_smb*beta_smb + cagr_hml*beta_hml

	
		if valuation_model == "DDM":
			growth_rate_dist,div = get_div_g_rate(id_instrument)

			if div == 0:
				mean_value = 0
			else:
				output_distributions = []
				for i in range(iterations):
					new_g = growth_rate_dist[i]		
					discount_factor = ke - new_g
					value_per_share = div*(1+new_g)/discount_factor
					value_per_share = value_per_share.values[0]
					output_distributions.append(value_per_share)
				
				perc_25 = np.percentile(output_distributions,25)
				mean_value = np.mean(output_distributions)
				plt.hist(output_distributions, bins = 100, density = False)
				plt.axvline(last_price, color='k', linestyle='dashed', linewidth=1)
				plt.xlim([0,40])
				plt.savefig(directory+r"\valuation_ddm_ff_{}".format(symbol))
			return mean_value
 
		elif valuation_model == "FCFE":
			df,fcf,shares_out,cash_per_share =company_cash_flows(3,0.23)
			fcfe = df["fcfe_per_share"].iloc[-1]
			value_per_share = fcfe*(1+0.02)/(ke-0.02)
			return value_per_share

		elif valuation_model == "FCF":
			df,fcf,shares_out,cash_per_share =company_cash_flows(3,0.23)
			fcf_per_share = fcf/shares_out
			wacc,debt = get_wacc(ke)
			debt_per_share = debt/shares_out
			value_per_share = (fcf_per_share*(1+0.02)/(wacc-0.02)) - debt_per_share + cash_per_share
			return value_per_share

		else:
			pass 

	else:
		pass




def peer_valuation(id_instrument):
	id_industry = pd.read_sql("SELECT ID_INDUSTRY FROM IndustryStockMerge WHERE ID_STOCK = {}".format(id_instrument),acc_engine)["ID_INDUSTRY"].values[0]
	df_industry_ratio = pd.read_sql("SELECT * FROM IndustryAverage WHERE ID_INDUSTRY = {}".format(id_industry),acc_engine)
	last_price = pd.read_sql("SELECT Data,Close FROM PriceData WHERE ID_STOCK= {}".format(id_instrument),acc_engine,index_col = "Data").sort_values("Data")["Close"].iloc[-1]
	pe_ratio = df_industry_ratio["pe_ratio_peer"].values[0]
	forward_pe =df_industry_ratio["forward_pe_ratio_peer"].values[0]
	peg_ratio =df_industry_ratio["peg_ratio_peer"].values[0]
	price_to_sales = df_industry_ratio["price_to_sales_peer"].values[0]
	pb_ratio = df_industry_ratio["price_to_book_peer"].values[0]
	p_to_cfo =df_industry_ratio["price_to_cfo_peer"].values[0]
	p_to_fcf =df_industry_ratio["price_to_fcf_peer"].values[0]

	df_ratio =pd.read_sql("SELECT * FROM RatioData WHERE ID_STOCK = {}".format(id_instrument),acc_engine)
	shares_out = df_ratio["shares_outstanding"].values[0]
	eps = df_ratio["eps"].values[0]
	cfo_per_share = df_ratio["cfo"].values[0]/ shares_out
	fcf_per_share =df_ratio["fcf"].values[0]/ shares_out
	sales_per_share = df_ratio["sales"].values[0] /shares_out
	book_value = df_ratio["book_value_per_share"].values[0]
	industry_growth_rate = np.round(pe_ratio/peg_ratio,2)

	value_pe = np.round(pe_ratio*eps,2)
	value_cfo = np.round(cfo_per_share*p_to_cfo)
	value_fcf = np.round(fcf_per_share*p_to_fcf)
	value_sales = np.round(sales_per_share*price_to_sales)
	value_book = np.round(pb_ratio*book_value)
	mean_value =np.mean([value_pe,value_cfo,value_fcf,value_sales,value_book])
	list_values = np.round([value_sales,value_book,value_pe,value_cfo,value_fcf,mean_value,industry_growth_rate],2)
	
	return list_values




def return_evaluation(id_instrument,_model):
	id_industry = pd.read_sql("SELECT ID_INDUSTRY FROM IndustryStockMerge WHERE ID_STOCK = {}".format(id_instrument),acc_engine)["ID_INDUSTRY"].values[0]
	

	df_price = pd.read_sql("SELECT Data,Close FROM PriceData WHERE ID_STOCK ={}".format(id_instrument),acc_engine,index_col = "Data").sort_values("Data")
	df_industry = pd.read_sql("SELECT Data,Close FROM BenchmarkPriceData WHERE ID_STOCK ={}".format(id_industry),acc_engine)
	df_market =pd.read_sql("SELECT Data,Close FROM BenchmarkPriceData WHERE ID_STOCK ={}".format(4),acc_engine)

	if beta_model == "Industry Fama and French":
		df_regression,df_returns = get_betas(id_instrument,beta_model,52,1)
		df_industry = df_industry.resample("W").last()
		df_industry["returns_industry"] = df_industry["Close"].pct_change()


	else:
		pass



def qr_regression_market(id_instrument):
	symbol = pd.read_sql("SELECT (SYMBOL) FROM StockIndex WHERE ID_STOCK= {}".format(id_instrument),acc_engine)["SYMBOL"].values[0]
	df_market = pd.read_sql("SELECT Data,Close FROM BenchmarkPriceData	 WHERE ID_BENCH	 = {}".format(4),acc_engine,index_col = "Data").sort_values("Data")
	df_price = pd.read_sql("SELECT Data,Close FROM PriceData WHERE ID_STOCK = {}".format(id_instrument),acc_engine,index_col = "Data").sort_values("Data")
	df_market.rename(columns = {"Close":"Close_market"},inplace = True)
	df_price.rename(columns = {"Close":"Close_fund"},inplace = True)
	df_merge = df_price.merge(df_market,how = "inner", right_index = True, left_index = True)

	df_merge = df_merge.resample("W").last()
	df_merge = df_merge.pct_change()
	df_merge = df_merge.iloc[1::]

	## OLS Model
	X = df_merge["Close_market"]
	X1  = sm.add_constant(X)
	model = sm.OLS(df_merge["Close_fund"],X1)
	ols = model.fit()
	ols_ci = ols.conf_int().loc["Close_market"].tolist()
	ols = dict(a=ols.params["const"], b=ols.params["Close_market"], lb=ols_ci[0], ub=ols_ci[1])
	get_y_ols = lambda a, b: a + b * X
	df_ols = pd.DataFrame.from_dict(ols,orient = "index").T

	## QR Model
	quantiles = np.arange(0.05, 0.96, 0.1)
	model = smf.quantreg('Close_fund ~ Close_market',df_merge)#.fit(q=0.5)
	def fit_model(q):
		res = model.fit(q=q)
		return [q, res.params["Intercept"], res.params["Close_market"]] + res.conf_int().loc["Close_market"].tolist()

	models = [fit_model(x) for x in quantiles]
	models = pd.DataFrame(models, columns=["Quantil", "Alpha", "Beta_Market", "lb", "ub"])
	fig, ax = plt.subplots(figsize=(8, 6))
	get_y = lambda Alpha, Beta_Market: Alpha + Beta_Market * X


	quantile_count = 0
	for i in range(models.shape[0]):
		quantile = models["Quantil"].iloc[quantile_count]
		quantile = quantile.round(2)
		y = get_y(models.Alpha[i], models.Beta_Market[i])
		if quantile == 0.05:
			ax.plot(X, y, color="red",label=f"Quantile: {quantile}")
		elif quantile == 0.95:
			ax.plot(X, y, color="green",label=f"Quantile: {quantile}")
		else:
			ax.plot(X, y, linestyle="dotted",color = "grey")
		quantile_count = quantile_count +1

	y_ols = get_y_ols(df_ols.a[0], df_ols.b[0])

	fig,ax = plt.subplots(1,figsize = (15,8))
	ax.plot(X, y_ols, color="blue", label="OLS")
	ax.scatter(df_merge.Close_market, df_merge.Close_fund, alpha=0.2)
	ax.set_title("Regressão Quantil vs. OLS ")
	ax.set_xlim((-0.10, 0.1))
	ax.set_ylim((-0.10, 0.1))
	legend = ax.legend()
	ax.set_xlabel("Retornos Mercado", fontsize=12)
	ax.set_ylabel("Retornos Fundo", fontsize=12)
	directory = r"YOUR DIRECTORY\{}".format(symbol)
	plt.tight_layout()
	plt.savefig(directory+r"\qr_regression_market_{}".format(symbol),dpi = 199)
	plt.close("all")
	df_qr = models
	return df_qr


