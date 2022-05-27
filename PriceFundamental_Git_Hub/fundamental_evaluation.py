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
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import operator
from numpy.linalg import solve
from scipy.stats import moment,norm,skew,kurtosis,skewnorm
from fitter import Fitter, get_common_distributions, get_distributions
from scipy import stats, integrate
import logging
logging.basicConfig(level = logging.DEBUG)  

conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=YOUR DIRECTORY\StockPriceFundamentalData.accdb;')
cnn_url = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
acc_engine = create_engine(cnn_url)


conn_str_funds = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=YOUR DIRECTORY\FundDataBase.accdb;')
cnn_url_funds = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str_funds)}"
acc_engine_funds = create_engine(cnn_url_funds)


def get_fundamental_ratios(id_stock):
	symbol = pd.read_sql("SELECT (Symbol) FROM StockIndex WHERE ID_STOCK = {}".format(id_stock),acc_engine)["Symbol"].values[0]
	directory = r"YOUR DIRECTORY\{}".format(symbol)

	df_balance = pd.read_sql("SELECT * FROM BalanceSheetData WHERE ID_STOCK = {} AND period_type = '{}' ".format(id_stock,"Quarter"),acc_engine,index_col = "Data").sort_values(by="Data")
	df_financials = pd.read_sql("SELECT * FROM IncomeStatementData WHERE ID_STOCK = {} AND period_type = '{}' ".format(id_stock,"Quarter"),acc_engine,index_col = "Data").sort_values(by="Data")
	df_cash_flow = pd.read_sql("SELECT * FROM CashFlowData WHERE ID_STOCK = {} AND period_type = '{}' ".format(id_stock,"Quarter"),acc_engine,index_col = "Data").sort_values(by="Data")
	
	df_financials["gross_margin"] = df_financials["grossProfit"]/df_financials["totalRevenue"]
	df_financials["operating_margin"] = df_financials["ebit"]/df_financials["totalRevenue"]
	df_financials["ni_margin"] = df_financials["netIncome"]/df_financials["totalRevenue"]


	mean_gross_margin = df_financials.gross_margin.mean()
	mean_operating_margin= df_financials.operating_margin.mean()
	mean_operating_margin = df_financials.ni_margin.mean()
	std_gross_margin = df_financials.gross_margin.std()
	std_operating_margin= df_financials.operating_margin.std()
	std_operating_margin = df_financials.ni_margin.std()
	df_cash_flow["cfo_margin"] = df_cash_flow["operatingCashflow"]/df_financials["totalRevenue"]
	df_balance["d/e"] = df_balance["shortLongTermDebtTotal"]/df_balance["totalShareholderEquity"]


	df_cash_flow["cash_flow_coverage_ratio"] = df_cash_flow["operatingCashflow"] / df_financials["interestAndDebtExpense"]
	df_cash_flow["cash_flow_coverage_prime"] = df_cash_flow["operatingCashflow"] / (df_cash_flow["cashflowFromFinancing"]*(-1))
	

	## Margins Graph

	barWidth = 0.25
	plt.bar(list(df_financials.index), df_financials["gross_margin"], width = barWidth,
        edgecolor ='grey', label ='Gross Margin')
	plt.bar(df_financials.index, df_financials["operating_margin"], width = barWidth,
        edgecolor ='grey', label ='Operating Margin')
	plt.bar(df_financials.index, df_financials["ni_margin"], width = barWidth,
        edgecolor ='grey', label ='NI Margin')
	plt.legend(loc = "best")
	df_financials["gross_margin"].plot()
	df_financials["operating_margin"].plot()
	df_financials["ni_margin"].plot()
	plt.axhline(0.20, color='k', linestyle='dashed', linewidth=1)
	plt.axhline(0.40, color='k', linestyle='dashed', linewidth=1)
	plt.axhline(0.60, color='k', linestyle='dashed', linewidth=1)
	plt.savefig(directory + r"\goi_margin_{}.png".format(symbol))
	plt.close("all")

	## Cfo Margin Graph
	plt.bar(df_cash_flow.index, df_cash_flow["cfo_margin"], width = 1,
        edgecolor ='grey', label ='CFO Margin')
	plt.axhline(0.20, color='k', linestyle='dashed', linewidth=1)
	plt.axhline(0.40, color='k', linestyle='dashed', linewidth=1)
	plt.axhline(0.60, color='k', linestyle='dashed', linewidth=1)
	plt.legend(loc = "best")
	plt.savefig(directory + r"\cfo_margin_{}.png".format(symbol))
	plt.close("all")


	## Coverage Ratios
	df_financials["coverage_ratio"] = df_financials["ebit"] / df_financials["interestAndDebtExpense"]
	df_financials["cash_flow_coverage_prime"] = df_cash_flow["operatingCashflow"] / (df_cash_flow["cashflowFromFinancing"]*(-1))
	df_financials["coverage_ratio"].plot()
	df_financials["cash_flow_coverage_prime"].plot()
	plt.legend(loc = "best")
	plt.savefig(directory + r"\coverage_ratio_{}.png".format(symbol))
	plt.close("all")

	## ROA and ROE 

	df_financials["roa"] = df_financials["netIncome"] / df_balance["totalAssets"]
	df_financials["roe"] = df_financials["netIncome"] / df_balance["totalShareholderEquity"]
	df_financials["roa"].plot()
	df_financials["roe"].plot()
	plt.legend(loc = "best")
	plt.savefig(directory + r"\roa_roe_{}.png".format(symbol))
	plt.close("all")



