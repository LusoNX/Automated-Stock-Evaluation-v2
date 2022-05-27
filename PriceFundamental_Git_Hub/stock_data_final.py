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

api_key = "Alpha Vantage API KEY"

conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=YOUR DIRECTORY\StockPriceFundamentalData.accdb;')
cnn_url = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
acc_engine = create_engine(cnn_url)
def make_request (method, function, symbol, api_key):
	if method == "GET":
		base_url = "https://www.alphavantage.co/query?"
		response = requests.get(base_url+"function=" +function +"&symbol="+symbol +"&apikey="+api_key)
		return response.json()

def get_news_calendar_recommendations_data(id_symbol,symbol,acc_engine):
	ticker = yf.Ticker(symbol)
	today_date = date.today()
	#news_data = ticker.news
	time.sleep(1)
	calendar = ticker.calendar
	analyst_recommendations = ticker.recommendations
	institutional_holders = ticker.institutional_holders
	sustainability = ticker.sustainability
	calendar = calendar.T

	try:
		sustainability = sustainability.T
	except:
		sustainability = 0.0
	
	try:
		esg_score = sustainability["totalEsg"].values[0]
	except:
		esg_score = 0.0

	try:
		esg_perf = sustainability["esgPerformance"].values[0]
	except:
		esg_perf = "Not_APP"

	
	# Earnings estimate calendar Data
	calendar.reset_index(inplace = True)
	calendar["ID_STOCK"] = id_symbol
	calendar.rename(columns = {"Earnings Date":"earnings_date"},inplace = True)
	calendar.rename(columns = {"Earnings Average":"earnings_estimate"},inplace = True)
	calendar.rename(columns = {"Earnings Low":"earnings_low"},inplace = True)
	calendar.rename(columns = {"Earnings High":"earnings_high"},inplace = True)
	calendar.rename(columns = {"Revenue Average":"rev_estimate"},inplace = True)
	calendar.rename(columns = {"Revenue Low":"rev_low"},inplace = True)
	calendar.rename(columns = {"Revenue High":"rev_high"},inplace = True)
	calendar.set_index("ID_STOCK",inplace = True)
	calendar = calendar.sort_values("earnings_date")
	calendar.drop(columns = ["index"],inplace = True)
	calendar[["earnings_estimate","earnings_low","earnings_high","rev_estimate","rev_low","rev_high"]] = calendar[["earnings_estimate","earnings_low","earnings_high","rev_estimate","rev_low","rev_high"]].astype(float)

	calendar_exists = pd.read_sql("SELECT * FROM EarningsEstimate WHERE ID_STOCK = {}".format(id_symbol),acc_engine)
	last_date_reported_1 = calendar["earnings_date"].iloc[-1].to_pydatetime()
	if calendar_exists.empty:
		calendar.to_sql("EarningsEstimate",acc_engine,if_exists = "append")
	else:
		database_date_reported_1 = calendar_exists["earnings_date"].iloc[-1].to_pydatetime()
		if database_date_reported_1 == last_date_reported_1:
			pass
		else:
			calendar.to_sql("EarningsEstimate",acc_engine,if_exists = "append")
			pass
		pass


	# Analyst Recommendations
	df_analyst = analyst_recommendations
	df_analyst.reset_index(inplace = True)
	df_analyst["ID_STOCK"] = id_symbol
	df_analyst.set_index("ID_STOCK",inplace = True)
	df_analyst.rename(columns = {"Date":"Data"},inplace = True)
	df_analyst.rename(columns = {"Firm":"firm"},inplace = True)
	df_analyst.rename(columns = {"From Grade":"previous_grade"},inplace = True)
	df_analyst.rename(columns = {"To Grade":"latest_grade"},inplace = True)
	df_analyst.rename(columns = {"Action":"action_update"},inplace = True)

	df_analyst_exist = pd.read_sql("SELECT * FROM AnalystRecommendation WHERE ID_STOCK = {} ".format(id_symbol),acc_engine, index_col = "ID_STOCK")
	if df_analyst_exist.empty: 
		df_analyst.to_sql("AnalystRecommendation",acc_engine,if_exists = "append")
	else:
		df_analyst_merge = df_analyst_exist.append(df_analyst).drop_duplicates(keep = False)
		df_analyst_merge.to_sql("AnalystRecommendation",acc_engine,if_exists = "append")


	## Institutional Holders 
	df_institutional = institutional_holders
	df_institutional.reset_index(inplace = True)
	df_institutional["ID_STOCK"] = id_symbol
	df_institutional.set_index("ID_STOCK",inplace = True)
	df_institutional.rename(columns = {"Holder":"holder"},inplace = True)
	df_institutional.rename(columns = {"Date Reported":"Date_reported"},inplace = True)
	df_institutional.rename(columns = {"% Out":"perc_held"},inplace = True)
	df_institutional = df_institutional[["holder","Date_reported","perc_held"]]
	last_date_reported = df_institutional["Date_reported"].iloc[-1]
	df_institutional_exist = pd.read_sql("SELECT * FROM InstitutionalHolders WHERE ID_STOCK = {} ".format(id_symbol),acc_engine)

	if df_institutional_exist.empty:
		df_institutional.to_sql("InstitutionalHolders",acc_engine,if_exists = "append")

	else:
		database_date_reported = df_institutional_exist["Date_reported"].iloc[-1]
		if database_date_reported == last_date_reported:
			pass
		else:
			df_institutional.to_sql("InstitutionalHolders",acc_engine,if_exists = "append")
			pass
		pass

	
	
	return esg_score,esg_perf




def benchmark_price_data(acc_engine):
	df_benchmark_index = pd.read_sql("IndustryIndex",acc_engine)
	for i,x in zip(df_benchmark_index["ID_INDUSTRY"],df_benchmark_index["Benchmark_Symbol"]):
		ticker = yf.Ticker(x)
		benchmark_type = pd.read_sql("SELECT (Benchmark_Type) FROM IndustryIndex WHERE ID_INDUSTRY= {}".format(i),acc_engine).values[0][0]
		price_data = ticker.history(period = "max")
		df_price = price_data
		df_price.reset_index(inplace = True)
		df_price["ID_BENCH"] = i
		df_price = df_price[["ID_BENCH","Date","Open","High","Low","Close"]]
		df_price[["Open","High","Low","Close"]] =df_price[["Open","High","Low","Close"]].astype(float)
		df_price.rename(columns = {"Date":"Data"},inplace = True)
		df_price = df_price.sort_values(by = "Data")
		df_price_exists = pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH = {}".format(i),acc_engine).sort_values(by = "Data")
		df_price.set_index("ID_BENCH",inplace = True)
		df_price["Type"] = benchmark_type


		if df_price_exists.empty:
			df_price.to_sql("BenchmarkPriceData",acc_engine,if_exists = "append")
		else:
			old_date = df_price_exists.iloc[-1]["Data"]
			new_date = df_price.iloc[-1]["Data"]
			#new_date = np.datetime64(new_date)
			#old_date = np.datetime64(old_date)


			if new_date > old_date:
				mask = (df_price["Data"] > old_date)
				df_price =df_price[mask]
				df_price.to_sql("BenchmarkPriceData",acc_engine,if_exists = "append")
			else:
				pass


def get_balance_sheet(symbol):
	r = make_request("GET","BALANCE_SHEET",symbol,api_key)
	r_annual = r["annualReports"]
	r_quarterly = r["quarterlyReports"]
	df_annual = json_normalize(r_annual)
	df_quarterly = json_normalize(r_quarterly)
	
	return df_annual,df_quarterly


def get_income_statement(symbol):
	r = make_request("GET","INCOME_STATEMENT",symbol,api_key)
	r_annual = r["annualReports"]
	r_quarterly = r["quarterlyReports"]
	df_annual = json_normalize(r_annual)
	df_quarterly = json_normalize(r_quarterly)
	
	return df_annual,df_quarterly




def get_cash_flow(symbol):
	r = make_request("GET","CASH_FLOW",symbol,api_key)
	r_annual = r["annualReports"]
	r_quarterly = r["quarterlyReports"]
	df_annual = json_normalize(r_annual)
	df_quarterly = json_normalize(r_quarterly)
	print(df_quarterly.columns)

	return df_annual,df_quarterly






def get_price_fundamental_data_data(id_symbol,symbol,_update_all,acc_engine):
	ticker = yf.Ticker(symbol)
	price_data = ticker.history(period = "max")
	df_price = price_data
	df_price.reset_index(inplace = True)
	df_price["ID_STOCK"] = id_symbol
	df_price = df_price[["ID_STOCK","Date","Open","High","Low","Close"]]
	df_price[["Open","High","Low","Close"]] =df_price[["Open","High","Low","Close"]].astype(float)
	df_price.rename(columns = {"Date":"Data"},inplace = True)
	df_price = df_price.sort_values(by = "Data")
	df_price_exists = pd.read_sql("SELECT * FROM PriceData WHERE ID_STOCK = {}".format(id_symbol),acc_engine).sort_values(by = "Data")
	df_price.set_index("ID_STOCK",inplace = True)

	if df_price_exists.empty:
		df_price.to_sql("PriceData",acc_engine,if_exists = "append")
	else:
		old_date = df_price_exists.iloc[-1]["Data"]
		new_date = df_price.iloc[-1]["Data"]

		if new_date > old_date:
			df_price = df_price.iloc[[-1]]
			df_price.to_sql("PriceData",acc_engine,if_exists = "append")
		else:
			pass


	if _update_all == True:
		df_financials,df_quarter_financials = get_income_statement(symbol)
		df_balance,df_quarter_balance = get_balance_sheet(symbol)
		df_cash_flow,df_quarter_cash_flow = get_cash_flow(symbol)
		df_balance.rename(columns ={"fiscalDateEnding":"Data"},inplace = True)
		df_quarter_balance.rename(columns ={"fiscalDateEnding":"Data"},inplace = True)
		df_financials.rename(columns ={"fiscalDateEnding":"Data"},inplace = True)
		df_quarter_financials.rename(columns ={"fiscalDateEnding":"Data"},inplace = True)
		df_cash_flow.rename(columns ={"fiscalDateEnding":"Data"},inplace = True)
		df_quarter_cash_flow.rename(columns ={"fiscalDateEnding":"Data"},inplace = True)
		df_balance = df_balance.sort_values("Data")
		df_quarter_balance = df_quarter_balance.sort_values("Data")
		df_financials = df_financials.sort_values("Data")
		df_quarter_financials = df_quarter_financials.sort_values("Data")
		df_cash_flow = df_cash_flow.sort_values("Data")
		df_quarter_cash_flow = df_quarter_cash_flow.sort_values("Data")

		df_balance = df_balance.replace("None",0)
		df_quarter_balance = df_quarter_balance.replace("None",0)
		df_financials = df_financials.replace("None",0)
		df_quarter_financials = df_quarter_financials.replace("None",0)
		df_cash_flow = df_cash_flow.replace("None",0)
		df_quarter_cash_flow = df_quarter_cash_flow.replace("None",0)
		df_quarter_balance[['totalAssets', 'totalCurrentAssets', 'cashAndCashEquivalentsAtCarryingValue', 'cashAndShortTermInvestments', 'inventory', 'currentNetReceivables', 'totalNonCurrentAssets', 'propertyPlantEquipment', 'accumulatedDepreciationAmortizationPPE', 'intangibleAssets', 'intangibleAssetsExcludingGoodwill', 'goodwill', 'investments', 'longTermInvestments', 'shortTermInvestments', 'otherCurrentAssets', 'otherNonCurrrentAssets', 'totalLiabilities', 'totalCurrentLiabilities', 'currentAccountsPayable', 'deferredRevenue', 'currentDebt', 'shortTermDebt', 'totalNonCurrentLiabilities', 'capitalLeaseObligations', 'longTermDebt', 'currentLongTermDebt', 'longTermDebtNoncurrent', 'shortLongTermDebtTotal', 'otherCurrentLiabilities', 'otherNonCurrentLiabilities', 'totalShareholderEquity', 'treasuryStock', 'retainedEarnings', 'commonStock', 'commonStockSharesOutstanding']] = df_quarter_balance[['totalAssets', 'totalCurrentAssets', 'cashAndCashEquivalentsAtCarryingValue', 'cashAndShortTermInvestments', 'inventory', 'currentNetReceivables', 'totalNonCurrentAssets', 'propertyPlantEquipment', 'accumulatedDepreciationAmortizationPPE', 'intangibleAssets', 'intangibleAssetsExcludingGoodwill', 'goodwill', 'investments', 'longTermInvestments', 'shortTermInvestments', 'otherCurrentAssets', 'otherNonCurrrentAssets', 'totalLiabilities', 'totalCurrentLiabilities', 'currentAccountsPayable', 'deferredRevenue', 'currentDebt', 'shortTermDebt', 'totalNonCurrentLiabilities', 'capitalLeaseObligations', 'longTermDebt', 'currentLongTermDebt', 'longTermDebtNoncurrent', 'shortLongTermDebtTotal', 'otherCurrentLiabilities', 'otherNonCurrentLiabilities', 'totalShareholderEquity', 'treasuryStock', 'retainedEarnings', 'commonStock', 'commonStockSharesOutstanding']].astype(float)
		df_balance[['totalAssets', 'totalCurrentAssets', 'cashAndCashEquivalentsAtCarryingValue', 'cashAndShortTermInvestments', 'inventory', 'currentNetReceivables', 'totalNonCurrentAssets', 'propertyPlantEquipment', 'accumulatedDepreciationAmortizationPPE', 'intangibleAssets', 'intangibleAssetsExcludingGoodwill', 'goodwill', 'investments', 'longTermInvestments', 'shortTermInvestments', 'otherCurrentAssets', 'otherNonCurrrentAssets', 'totalLiabilities', 'totalCurrentLiabilities', 'currentAccountsPayable', 'deferredRevenue', 'currentDebt', 'shortTermDebt', 'totalNonCurrentLiabilities', 'capitalLeaseObligations', 'longTermDebt', 'currentLongTermDebt', 'longTermDebtNoncurrent', 'shortLongTermDebtTotal', 'otherCurrentLiabilities', 'otherNonCurrentLiabilities', 'totalShareholderEquity', 'treasuryStock', 'retainedEarnings', 'commonStock', 'commonStockSharesOutstanding']] =df_balance[['totalAssets', 'totalCurrentAssets', 'cashAndCashEquivalentsAtCarryingValue', 'cashAndShortTermInvestments', 'inventory', 'currentNetReceivables', 'totalNonCurrentAssets', 'propertyPlantEquipment', 'accumulatedDepreciationAmortizationPPE', 'intangibleAssets', 'intangibleAssetsExcludingGoodwill', 'goodwill', 'investments', 'longTermInvestments', 'shortTermInvestments', 'otherCurrentAssets', 'otherNonCurrrentAssets', 'totalLiabilities', 'totalCurrentLiabilities', 'currentAccountsPayable', 'deferredRevenue', 'currentDebt', 'shortTermDebt', 'totalNonCurrentLiabilities', 'capitalLeaseObligations', 'longTermDebt', 'currentLongTermDebt', 'longTermDebtNoncurrent', 'shortLongTermDebtTotal', 'otherCurrentLiabilities', 'otherNonCurrentLiabilities', 'totalShareholderEquity', 'treasuryStock', 'retainedEarnings', 'commonStock', 'commonStockSharesOutstanding']].astype(float)
		
		df_quarter_financials[list(df_quarter_financials.columns)[2::]] = df_quarter_financials[list(df_quarter_financials.columns)[2::]].astype(float)
		df_financials[list(df_financials.columns)[2::]] = df_financials[list(df_financials.columns)[2::]].astype(float)
		df_quarter_financials["Data"] = pd.to_datetime(df_quarter_financials["Data"])
		df_financials["Data"] = pd.to_datetime(df_financials["Data"])
		df_cash_flow = df_cash_flow[["Data","operatingCashflow","capitalExpenditures","cashflowFromInvestment","cashflowFromFinancing","dividendPayout"]]
		df_quarter_cash_flow = df_quarter_cash_flow[["Data","operatingCashflow","capitalExpenditures","cashflowFromInvestment","cashflowFromFinancing","dividendPayout"]]
		df_quarter_cash_flow[list(df_quarter_cash_flow.columns)[1::]] = df_quarter_cash_flow[list(df_quarter_cash_flow.columns)[1::]].astype(float)
		df_cash_flow[list(df_cash_flow.columns)[1::]] = df_cash_flow[list(df_cash_flow.columns)[1::]].astype(float)
		df_quarter_cash_flow["Data"] = pd.to_datetime(df_quarter_cash_flow["Data"])
		df_cash_flow["Data"] = pd.to_datetime(df_cash_flow["Data"])



		df_quarter_financials["period_type"] = "Quarter"
		df_financials["period_type"] = "Annual"
		df_quarter_balance["period_type"] = "Quarter"
		df_balance["period_type"] = "Annual"
		df_quarter_cash_flow["period_type"] = "Quarter"
		df_cash_flow["period_type"] = "Annual"


		df_quarter_financials["ID_STOCK"] = id_symbol
		df_financials["ID_STOCK"] = id_symbol
		df_quarter_balance["ID_STOCK"] = id_symbol
		df_balance["ID_STOCK"] = id_symbol
		df_quarter_cash_flow["ID_STOCK"] = id_symbol
		df_cash_flow["ID_STOCK"] = id_symbol


		df_balance["Data"]= pd.to_datetime(df_balance["Data"])
		df_quarter_balance["Data"] = pd.to_datetime(df_quarter_balance["Data"])
		

		df_quarter_financials.set_index("ID_STOCK",inplace = True)
		df_financials.set_index("ID_STOCK",inplace = True)
		df_quarter_balance.set_index("ID_STOCK",inplace = True)
		df_balance.set_index("ID_STOCK",inplace = True)
		df_quarter_cash_flow.set_index("ID_STOCK",inplace = True)
		df_cash_flow.set_index("ID_STOCK",inplace = True)
		df_balance.drop(columns	 ="reportedCurrency",inplace = True)
		df_quarter_balance.drop(columns	 ="reportedCurrency",inplace = True)



		df_balance_exists_annual = pd.read_sql("SELECT * FROM BalanceSheetData WHERE ID_STOCK ={} AND period_type = '{}' ".format(id_symbol,"Annual"),acc_engine,index_col = "ID_STOCK")
		df_balance_exists_quarter = pd.read_sql("SELECT * FROM BalanceSheetData WHERE ID_STOCK ={} AND period_type = '{}' ".format(id_symbol,"Quarter"),acc_engine ,index_col = "ID_STOCK")
		df_balance_exists_annual = df_balance_exists_annual.sort_values(by = "Data")
		df_balance_exists_quarter =df_balance_exists_quarter.sort_values(by = "Data")

		cash_flow_columns = list(pd.read_sql("SELECT * FROM CashFlowData WHERE ID_STOCK ={} AND period_type = '{}' ".format(id_symbol,"Annual"),acc_engine,index_col = "ID_STOCK"))
		df_cash_flow = df_cash_flow[cash_flow_columns]
		df_quarter_cash_flow = df_quarter_cash_flow[cash_flow_columns]
		if df_balance_exists_annual.empty:
			df_balance.to_sql("BalanceSheetData",acc_engine, if_exists = "append")
			df_financials.to_sql("IncomeStatementData",acc_engine, if_exists = "append")
			df_cash_flow.to_sql("CashFlowData",acc_engine, if_exists = "append")
		else:
			last_date_reported_balance_sheet_quarter =df_balance_exists_quarter.iloc[-1]["Data"]
			last_date_reported_balance_sheet_annual = df_balance_exists_annual.iloc[-1]["Data"]
			if last_date_reported_balance_sheet_quarter < df_quarter_balance.iloc[-1]["Data"]:
				for i in df_balance_exists_quarter["Data"]:
					df_quarter_balance = df_quarter_balance[df_quarter_balance['Data'] != i]
					df_quarter_financials = df_quarter_financials[df_quarter_financials['Data'] != i]
					df_quarter_cash_flow = df_quarter_cash_flow[df_quarter_cash_flow['Data'] != i]

				df_quarter_balance.to_sql("BalanceSheetData",acc_engine, if_exists = "append")
				df_quarter_financials.to_sql("IncomeStatementData",acc_engine, if_exists = "append")
				df_quarter_cash_flow.to_sql("CashFlowData",acc_engine, if_exists = "append")

			else:
				pass

			if last_date_reported_balance_sheet_annual < df_balance.iloc[-1]["Data"]:
				for i in df_balance_exists_annual["Data"]:
			
					df_balance = df_balance[df_balance['Data'] != i]
					df_financials = df_financials[df_financials['Data'] != i]
					df_cash_flow = df_cash_flow[df_cash_flow['Data'] != i]

				df_balance.to_sql("BalanceSheetData",acc_engine, if_exists = "append")
				df_financials.to_sql("IncomeStatementData",acc_engine, if_exists = "append")
				df_cash_flow.to_sql("CashFlowData",acc_engine, if_exists = "append")
				
			else:
				pass
			pass

		if df_balance_exists_quarter.empty:
			df_quarter_balance.to_sql("BalanceSheetData",acc_engine, if_exists = "append")
			df_quarter_financials.to_sql("IncomeStatementData",acc_engine, if_exists = "append")
			df_quarter_cash_flow.to_sql("CashFlowData",acc_engine, if_exists = "append")
		else:
			pass

	else:
		pass

def get_statements_data(id_symbol,acc_engine):
	df_balance = pd.read_sql("SELECT * FROM BalanceSheetData WHERE ID_STOCK = {} ".format(id_symbol),acc_engine,index_col = "ID_STOCK").sort_values(by="Data")
	df_financials = pd.read_sql("SELECT * FROM IncomeStatementData WHERE ID_STOCK = {} ".format(id_symbol),acc_engine,index_col = "ID_STOCK").sort_values(by="Data")
	df_cash_flow = pd.read_sql("SELECT * FROM CashFlowData WHERE ID_STOCK = {} ".format(id_symbol),acc_engine,index_col = "ID_STOCK").sort_values(by="Data")
	df_price = pd.read_sql("SELECT * FROM PriceData WHERE ID_STOCK = {}".format(id_symbol),acc_engine).sort_values(by="Data")

	last_price = df_price["Close"].iloc[-1]
	df_merge_index = pd.merge(df_balance, df_financials, on=['Data','period_type'])
	df_merge_index = pd.merge(df_merge_index,df_cash_flow, on=['Data','period_type'])

	df_merge_index["debt_to_equity"] = (df_merge_index["shortLongTermDebtTotal"]) /df_merge_index["totalShareholderEquity"]
	df_merge_index["cash_to_assets"] = (df_merge_index["cashAndShortTermInvestments"]) / df_merge_index["totalAssets"]
	df_merge_index["current_ratio"] = df_merge_index["totalCurrentAssets"] / df_merge_index["totalLiabilities"]
	df_merge_index["gross_margin"] = df_merge_index["grossProfit"] / df_merge_index["totalRevenue"]
	df_merge_index["operating_margin"] = df_merge_index["ebit"] /df_merge_index["totalRevenue"]
	df_merge_index["ni_margin"] = df_merge_index["netIncome"] / df_merge_index["totalRevenue"]
	df_merge_index["coverage_ratio"] = df_merge_index["ebit"] / df_merge_index["interestAndDebtExpense"]
	df_merge_index["roa"] = df_merge_index["netIncome"] / df_merge_index["totalAssets"]
	df_merge_index["roe"] = df_merge_index["netIncome"] / df_merge_index["totalShareholderEquity"]

	df_merge_index["cash_flow_coverage_ratio"] = df_merge_index["operatingCashflow"] / df_merge_index["interestAndDebtExpense"]
	df_merge_index["cash_flow_coverage_prime"] = df_merge_index["operatingCashflow"] / (df_merge_index["cashflowFromFinancing"]*(-1))
	df_final = df_merge_index[["Data","debt_to_equity","cash_to_assets","current_ratio","gross_margin","operating_margin","ni_margin","coverage_ratio","roa","roe","cash_flow_coverage_ratio","cash_flow_coverage_prime","operatingCashflow","interestAndDebtExpense","period_type"]]
	
	return df_final


def get_info_data(id_symbol,symbol,_df_statements,_esg_score,_esg_perf,acc_engine):
	df_statements = _df_statements.copy()
	ticker = yf.Ticker(symbol)
	info = ticker.info
	number_of_analyst = info["numberOfAnalystOpinions"]
	target_mean = info["targetMedianPrice"]
	target_median = info["targetMeanPrice"]
	eps =info["trailingEps"]
	forward_eps = info["forwardEps"]
	book_value_per_share = info["bookValue"]
	sales = info["totalRevenue"]
	try:
		trailing_PE = info["trailingPE"]
	except:
		trailing_PE = 0.0
	forward_PE = info["forwardPE"]
	short_ratio = info["shortRatio"]
	peg_ratio = info["pegRatio"]
	shares_outstanding = info["sharesOutstanding"]
	ev_to_ebitda = info["enterpriseToEbitda"]
	price_to_sales = info["priceToSalesTrailing12Months"]
	price_to_book = info["priceToBook"]
	insiders_holders_perc = info["heldPercentInsiders"]
	institutional_holders_perc = info["heldPercentInstitutions"]
	currency = info["currency"]
	country = info["country"]
	business_summary = info["longBusinessSummary"]
	category = info["category"]
	div_rate = info["dividendYield"]
	fcf = info["freeCashflow"]
	today_date = date.today()
	

	debt_to_equity = df_statements["debt_to_equity"].iloc[-1]
	cash_to_assets = df_statements["cash_to_assets"].iloc[-1]
	roa = df_statements["roa"].iloc[-1]
	roe = df_statements["roe"].iloc[-1]
	gross_margin = df_statements["gross_margin"].iloc[-1]
	operating_margin = df_statements["operating_margin"].iloc[-1]
	ni_margin = df_statements["ni_margin"].iloc[-1]
	try:
		if df_statements["interestAndDebtExpense"].iloc[-1] > 0:
			coverage_ratio = df_statements["coverage_ratio"].iloc[-1]
		else:
			coverage_ratio = df_statements["coverage_ratio"].iloc[-1]*(-1)
	except:
		coverage_ratio = 0.0

	cash_flow_coverage_prime = df_statements["cash_flow_coverage_prime"].iloc[-1]
	cfo =df_statements[df_statements["period_type"] == "Quarter"]["operatingCashflow"].iloc[-4:].sum()
	cfo_per_share = cfo/shares_outstanding
	df_price = pd.read_sql("SELECT * FROM PriceData WHERE ID_STOCK ={} ".format(id_symbol),acc_engine)
	price = df_price["Close"].iloc[-1]
	price_to_cfo = price / cfo_per_share
	price_to_fcf = price/(fcf/shares_outstanding)
	shares_outstanding = float(shares_outstanding)
	cfo = float(cfo)
	fcf = float(fcf)
	sales = float(sales)

	if div_rate == None:
		div_rate = 0.0
	else:
		pass
	list_of_values = [[id_symbol,today_date,eps,trailing_PE,forward_PE,
	peg_ratio,price_to_book,price_to_sales,price_to_cfo,
	price_to_fcf,div_rate,ev_to_ebitda,debt_to_equity,cash_to_assets,
	roa,roe,gross_margin,operating_margin,ni_margin,coverage_ratio,cash_flow_coverage_prime,
	short_ratio,number_of_analyst,target_mean,target_median,insiders_holders_perc,
	institutional_holders_perc,_esg_score,_esg_perf,book_value_per_share,cfo,fcf,sales,shares_outstanding]]
	df_ratio_data = pd.DataFrame(list_of_values,columns = ["ID_STOCK","Data_Appended","eps","pe_ratio","forward_pe_ratio","peg_ratio",
		"price_to_book_ratio","price_to_sales","price_to_cfo","price_to_fcf","div_rate","ev_to_ebitda","debt_to_equity","cash_to_assets","roa",
		"roe","gross_margin","operating_margin","ni_margin","coverage_ratio","coverage_ratio_prime","short_ratio","number_of_analyst","target_mean",
		"target_median","insider_holder_perc","institutional_holder_perc","esg_score","esg_performance","book_value_per_share","cfo","fcf","sales","shares_outstanding"])
	df_ratio_data["Data_Appended"] = pd.to_datetime(df_ratio_data["Data_Appended"])

	df_ratio_data.set_index("ID_STOCK",inplace = True)
	df_ratio_data_exists = pd.read_sql("SELECT * FROM RatioData WHERE ID_STOCK = {}".format(id_symbol),acc_engine)

	if df_ratio_data_exists.empty:
		df_ratio_data.to_sql("RatioData",acc_engine,if_exists = "append")
	else:
		last_date_reported = df_ratio_data_exists["Data_Appended"].iloc[-1]
		if last_date_reported >= today_date:
			pass 
		else:
			df_ratio_data.to_sql("RatioData",acc_engine,if_exists = "append")

def max_dd(DF):
	"function to calculate max drawdown"
	df = DF.copy()
	df["cum_return"] = (1 + df["Returns"]).cumprod()
	df["cum_roll_max"] = df["cum_return"].cummax()
	df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
	df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
	max_dd = df["drawdown_pct"].max()
	return max_dd

def get_stock_stats(id_stock,acc_engine):
	df_price = pd.read_sql("SELECT * FROM PriceData WHERE ID_STOCK ={}".format(id_stock),acc_engine).sort_values(by="Data")
	df_price.set_index("Data",inplace = True)
	df_price_d = df_price.copy()
	df_price_d["Returns"] = df_price_d["Close"].pct_change()
	df_price_d.dropna(inplace = True)

	#Daily Data
	skew_d = round(df_price_d["Returns"].skew(),4)
	kurt_d = round(df_price_d["Returns"].kurtosis(),4)
	df_sorted_d = df_price_d.sort_values(by = ["Returns"])
	df_var_5_d = df_sorted_d[df_sorted_d['Returns'].le(df_sorted_d['Returns'].quantile(0.05))]
	var_5_d = round(df_sorted_d.Returns.quantile(0.05),4)
	cvar_5_d = round(df_var_5_d["Returns"].mean(),4)
	worst_return_d = round(df_sorted_d.iloc[0:1].Returns.values[0],4)
	best_return_d = round(df_sorted_d.iloc[-1:].Returns.values[0],4)

	#Monthly data 
	df_price_m = df_price.copy()
	df_price_m = df_price_m.resample("M").last()
	df_price_m["Returns"] = df_price_m["Close"].pct_change()
	df_price_m.dropna(inplace = True)
	skew_m = round(df_price_m["Returns"].skew(),4)
	kurt_m = round(df_price_m["Returns"].kurtosis(),4)
	df_sorted_m = df_price_m.sort_values(by = ["Returns"])
	df_var_5_m = df_sorted_m[df_sorted_m['Returns'].le(df_sorted_m['Returns'].quantile(0.05))]
	var_5_m = round(df_sorted_m.Returns.quantile(0.05),4)
	cvar_5_m = round(df_var_5_m["Returns"].mean(),4)
	worst_return_m = round(df_sorted_m.iloc[0:1].Returns.values[0],4)
	best_return_m = round(df_sorted_m.iloc[-1:].Returns.values[0],4)

	# Yearly Data
	df_price_y = df_price.copy()
	df_price_y = df_price_y.resample("Y").last()
	df_price_y["Returns"] = df_price_y["Close"].pct_change()
	df_price_y.dropna(inplace = True)
	skew_y = round(df_price_y["Returns"].skew(),4)
	kurt_y = round(df_price_y["Returns"].kurtosis(),4)
	fund_kurtosis_y = round(df_price_y["Returns"].kurtosis(),4)
	df_sorted_y = df_price_y.sort_values(by = ["Returns"])
	df_var_5_y = df_sorted_y[df_sorted_y['Returns'].le(df_sorted_y['Returns'].quantile(0.05))]
	var_5_y = round(df_sorted_y.Returns.quantile(0.05),4)
	cvar_5_y = round(df_var_5_y["Returns"].mean(),4)
	worst_return_y = round(df_sorted_y.iloc[0:1].Returns.values[0],4)
	best_return_y = round(df_sorted_y.iloc[-1:].Returns.values[0],4)


	def create_table(_stock_skew,_stock_kurt,_var_5,_cvar_5,_worst_return,_best_return):
		list_of_values = [_stock_skew,_stock_kurt,_var_5,_cvar_5,_worst_return,_best_return]
		column_names = ["Skew","Kurtosis","Var 5","CVar 5","Pior Retorno","Melhor Retorno"]
		df_table= pd.DataFrame([list_of_values],columns = column_names)
		return df_table


	df_table_d = create_table(skew_d,kurt_d,var_5_d,cvar_5_d,worst_return_d,best_return_d)
	df_table_m = create_table(skew_m,kurt_m,var_5_m,cvar_5_m,worst_return_m,best_return_m)
	df_table_a = create_table(skew_y,kurt_y,var_5_y,cvar_5_y,worst_return_y,best_return_y)
	list_final_table = []
	list_of_tables = [df_table_d,df_table_m,df_table_a]

	list_of_names = ["Diário","Mensal","Anual"]
	for i in list_of_tables:
		list_final_table.append(i)
	df_table_final = pd.concat(list_final_table)
	df_table_final["TimeFrame"] = list_of_names
	df_table_final.set_index("TimeFrame",inplace=True)
	df_table_final[["Var 5","CVar 5","Pior Retorno","Melhor Retorno"]] =(df_table_final[["Var 5","CVar 5","Pior Retorno","Melhor Retorno"]]*100).round(2)

	# Montlhy returns
	f, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))#
	sns.distplot(df_price_m["Returns"] ,fit=norm, color="skyblue", ax = axes[0])
	stats.probplot(df_price_m['Returns'], plot=plt, fit = norm)
	axes[0].set_title('Returns Distribution', fontsize = 15, loc='center')
	axes[0].set_xlabel('Returns', fontsize = 13)
	axes[1].set_xlabel("Returns Probability",fontsize = 13)
	axes[1].yaxis.tick_right()
	plt.show()
	plt.close()
	return df_table_final



def outperformance_data(id_instrument,acc_engine):
	id_industry = pd.read_sql("SELECT (ID_INDUSTRY) FROM IndustryStockMerge WHERE ID_STOCK = {}".format(id_instrument),acc_engine).values[0][0]
	df_benchmark = pd.read_sql("SELECT * FROM BenchmarkPriceData WHERE ID_BENCH = {}".format(id_industry),acc_engine).sort_values(by = "Data")
	df_stock =pd.read_sql("SELECT * FROM PriceData WHERE ID_STOCK = {}".format(id_instrument),acc_engine).sort_values(by = "Data")
	df_benchmark.set_index("Data",inplace = True)
	df_stock.set_index("Data",inplace = True)


	df_values = df_benchmark.merge(df_stock,how = "inner", right_index = True, left_index = True)
	df_values = df_values[["Close_x","Close_y"]]
	df_values.rename(columns ={"Close_x":"Close_benchmark","Close_y":"Close_Fund"},inplace = True)


	#Daily 
	df_values_d = df_values.pct_change()
	df_values_d["outperformance_returns"] = df_values_d["Close_Fund"]-df_values_d["Close_benchmark"]
	df_values_d.dropna(inplace = True)
	last_return_d = df_values_d.iloc[-1]["outperformance_returns"]

	df_values_sorted_d = df_values_d.sort_values(by = ["outperformance_returns"])
	max_d = df_values_sorted_d["outperformance_returns"].max()
	min_d = df_values_sorted_d["outperformance_returns"].min()
	mean_d = df_values_sorted_d["outperformance_returns"].mean()
	p25_d = round(df_values_sorted_d.outperformance_returns.quantile(0.25),4)
	p50_d = round(df_values_sorted_d.outperformance_returns.quantile(0.50),4)
	p75_d =round(df_values_sorted_d.outperformance_returns.quantile(0.50),4)
	d_position_quantile =round(stats.percentileofscore(df_values_sorted_d['outperformance_returns'],last_return_d),4)

	# Monthly
	df_values_m = df_values.resample("M").last()
	df_values_m = df_values_m.pct_change()
	df_values_m.dropna(inplace = True)
	df_values_m["outperformance_returns"] = df_values_m["Close_Fund"]-df_values_m["Close_benchmark"]
	last_return_m = df_values_m.iloc[-1]["outperformance_returns"]
	df_values_sorted_m = df_values_m.sort_values(by = ["outperformance_returns"])
	max_m = df_values_sorted_m["outperformance_returns"].max()
	min_m = df_values_sorted_m["outperformance_returns"].min()
	mean_m = df_values_sorted_m["outperformance_returns"].mean()
	p25_m = round(df_values_sorted_m.outperformance_returns.quantile(0.25),4)
	p50_m = round(df_values_sorted_m.outperformance_returns.quantile(0.50),4)
	p75_m =round(df_values_sorted_m.outperformance_returns.quantile(0.50),4)
	month_position_quantile =round(stats.percentileofscore(df_values_sorted_m['outperformance_returns'],last_return_m),4)


	# Yearl 
	df_values_y = df_values.resample("Y").last()
	df_values_y = df_values_y.pct_change()
	df_values_y.dropna(inplace = True)
	df_values_y["outperformance_returns"] = df_values_y["Close_Fund"]-df_values_y["Close_benchmark"]
	last_return_y = df_values_y.iloc[-1]["outperformance_returns"]
	df_values_sorted_y = df_values_y.sort_values(by = ["outperformance_returns"])
	max_y = df_values_sorted_y["outperformance_returns"].max()
	min_y = df_values_sorted_y["outperformance_returns"].min()
	mean_y = df_values_sorted_y["outperformance_returns"].mean()
	p25_y = round(df_values_sorted_y.outperformance_returns.quantile(0.25),4)
	p50_y = round(df_values_sorted_y.outperformance_returns.quantile(0.50),4)
	p75_y =round(df_values_sorted_y.outperformance_returns.quantile(0.50),4)
	year_position_quantile =round(stats.percentileofscore(df_values_sorted_y['outperformance_returns'],last_return_y),4)


	def create_table_2(actual,percentil_pos,_min,p25,p50,p75,mean,_max):
		list_of_values = [actual,percentil_pos,_min,p25,p50,p75,mean,_max]
		column_names = ["Retorno Corrente","Percentil Posiçao","Minimo","Percentil 25","Percentil 50","Percentil 75","Media","Max"]
		df_table= pd.DataFrame([list_of_values],columns = column_names)
		return df_table

	
	df_table_d = create_table_2(last_return_d,d_position_quantile,min_d,p25_d,p50_d,p75_d,mean_d,max_d)

	df_table_m = create_table_2(last_return_m,month_position_quantile,min_m,p25_m,p50_m,p75_m,mean_m,max_m)
	df_table_a = create_table_2(last_return_y,year_position_quantile,min_y,p25_y,p50_y,p75_y,mean_y,max_y)
	list_final_table = []
	list_of_tables = [df_table_d,df_table_m,df_table_a]

	list_of_names = ["Retornos Diários","Retornos Mensais","Retornos Anuais"]
	for i in list_of_tables:
		list_final_table.append(i)
	df_table_final = pd.concat(list_final_table)
	df_table_final["TimeFrame"] = list_of_names
	df_table_final.set_index("TimeFrame",inplace=True)

def get_industry_average_data(acc_engine):
	df_stock_index = pd.read_sql("StockIndex",acc_engine)
	all_id_industry = list(pd.read_sql("IndustryStockMerge",acc_engine)["ID_INDUSTRY"])
	all_id_industry = list(set(all_id_industry))


	for i in all_id_industry:
		id_stocks = list(pd.read_sql("SELECT ID_STOCK FROM IndustryStockMerge WHERE ID_INDUSTRY = {}".format(i),acc_engine)["ID_STOCK"])
		df_ratio = pd.read_sql("RatioData",acc_engine)
		df_ratio_merge= df_ratio[df_ratio.ID_STOCK.isin(id_stocks)]
		today_date = df_ratio_merge["Data_Appended"].iloc[-1]
		today_date = today_date.to_pydatetime()
		df_ratio_merge.drop(columns = ["ID_STOCK","eps","number_of_analyst","target_mean","target_median","insider_holder_perc","institutional_holder_perc","esg_performance","fcf","cfo","shares_outstanding","book_value_per_share","sales"],inplace = True)
		df_ratio_average = df_ratio_merge.mean(axis=0)

		df_ratio_average = pd.DataFrame(df_ratio_average).T
		df_ratio_average["ID_INDUSTRY"] = i
		df_ratio_average["Data_Appended"] = [today_date]
		df_ratio_average.set_index("ID_INDUSTRY",inplace = True)

		new_columns_name = ['pe_ratio_peer', 'forward_pe_ratio_peer', 'peg_ratio_peer', 'price_to_book_peer',
		'price_to_sales_peer', 'price_to_cfo_peer', 'price_to_fcf_peer', 'div_rate_peer',
		'ev_to_ebitda_peer', 'debt_to_equity_peer', 'cash_to_assets_peer', 'roa_peer', 'roe_peer',
		'gross_margin_peer', 'operating_margin_peer', 'ni_margin_peer', 'coverage_ratio_peer',
		'coverage_ratio_prime_peer', 'short_ratio_peer', 'esg_score_peer',"Data_Appended"]
		df_ratio_average.columns = new_columns_name
		df_industry_average_exists = pd.read_sql("SELECT * FROM IndustryAverage WHERE ID_INDUSTRY = {}".format(i),acc_engine)
		df_industry_average_exists = df_industry_average_exists[df_industry_average_exists == today_date]
		if df_industry_average_exists.empty:
			df_ratio_average.to_sql("IndustryAverage",acc_engine,if_exists = "append")

		
		else:
			pass 



def data_append_sql(_update_all):
	df_symbols = pd.read_sql("StockIndex",acc_engine).sort_values(by="ID_STOCK")


	for i,x in zip(df_symbols["ID_STOCK"],df_symbols["SYMBOL"]):
		get_price_fundamental_data_data(i,x,_update_all,acc_engine)	#	get_price_fundamental_data_data(x)
		time.sleep(62)
		df_statements = get_statements_data(i,acc_engine)
		time.sleep(random.randint(1,2))
		esg_score,esg_perf = get_news_calendar_recommendations_data(i,x,acc_engine)			#time.sleep(random.randint(1,2))
		get_info_data(i,x,df_statements,esg_score,esg_perf,acc_engine)# esta linha de codigo tem que vir sempre depois da anterior 
		time.sleep(random.randint(1,6))
		benchmark_price_data(acc_engine)
		outperformance_data(i,acc_engine)
	
	get_industry_average_data(acc_engine)

data_append_sql(True) # true to update the financial statements only 


