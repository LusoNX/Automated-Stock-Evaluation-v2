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
import fund_evaluation
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



def fleishman(b, c, d):
	"""calculate the variance, skew and kurtois of a Fleishman distribution
	F = -c + bZ + cZ^2 + dZ^3, where Z ~ N(0,1)
	"""
	b2 = b * b
	c2 = c * c
	d2 = d * d
	bd = b * d
	var = b2 + 6*bd + 2*c2 + 15*d2
	skew = 2 * c * (b2 + 24*bd + 105*d2 + 2)
	kurt = 24 * (bd + c2 * (1 + b2 + 28*bd) + 
				 d2 * (12 + 48*bd + 141*c2 + 225*d2))
	return (var, skew, kurt)

def flfunc(b, c, d, skew, kurtosis):
	"""
	Given the fleishman coefficients, and a target skew and kurtois
	this function will have a root if the coefficients give the desired skew and kurtosis
	"""
	x,y,z = fleishman(b,c,d)
	return (x - 1, y - skew, z - kurtosis)

def flderiv(b, c, d):
	"""
	The deriviative of the flfunc above
	returns a matrix of partial derivatives
	"""
	b2 = b * b
	c2 = c * c
	d2 = d * d
	bd = b * d
	df1db = 2*b + 6*d
	df1dc = 4*c
	df1dd = 6*b + 30*d
	df2db = 4*c * (b + 12*d)
	df2dc = 2 * (b2 + 24*bd + 105*d2 + 2)
	df2dd = 4 * c * (12*b + 105*d)
	df3db = 24 * (d + c2 * (2*b + 28*d) + 48 * d**3)
	df3dc = 48 * c * (1 + b2 + 28*bd + 141*d2)
	df3dd = 24 * (b + 28*b * c2 + 2 * d * (12 + 48*bd + 
				  141*c2 + 225*d2) + d2 * (48*b + 450*d))
	return np.matrix([[df1db, df1dc, df1dd],
					  [df2db, df2dc, df2dd],
					  [df3db, df3dc, df3dd]])

def newton(a, b, c, skew, kurtosis, max_iter=25, converge=1e-5):
	"""Implements newtons method to find a root of flfunc."""
	f = flfunc(a, b, c, skew, kurtosis)
	for i in range(max_iter):
		if max(map(abs, f)) < converge:
			break
		J = flderiv(a, b, c)
		delta = -solve(J, f)
		(a, b, c) = delta + (a,b,c)
		f = flfunc(a, b, c, skew, kurtosis)
	return (a, b, c)


def fleishmanic(skew, kurt):
	"""Find an initial estimate of the fleisman coefficients, to feed to newtons method"""
	c1 = 0.95357 - 0.05679 * kurt + 0.03520 * skew**2 + 0.00133 * kurt**2
	c2 = 0.10007 * skew + 0.00844 * skew**3
	c3 = 0.30978 - 0.31655 * c1
	logging.debug("inital guess {},{},{}".format(c1,c2,c3))
	return (c1, c2, c3)


def fit_fleishman_from_sk(skew, kurt):
	"""Find the fleishman distribution with given skew and kurtosis
	mean =0 and stdev =1
	
	Returns None if no such distribution can be found
	"""
	if kurt < -1.13168 + 1.58837 * skew**2:
		return None
	a, b, c = fleishmanic(skew, kurt)
	coef = newton(a, b, c, skew, kurt)
	return(coef)

def fit_fleishman_from_standardised_data(data):
	"""Fit a fleishman distribution to standardised data."""
	skew = moment(data,3)
	kurt = moment(data,4)
	coeff = fit_fleishman_from_sk(skew,kurt)
	return coeff



def describe(data):
	"""Return summary statistics of as set of data"""
	mean = sum(data)/len(data)
	var = moment(data,2)
	skew = moment(data,3)/var**1.5
	kurt = moment(data,4)/var**2
	return (mean,var,skew,kurt)

def generate_fleishman(a,b,c,d,N=100):
	"""Generate N data items from fleishman's distribution with given coefficents"""
	Z = norm.rvs(size=N)
	F = a + Z*(b +Z*(c+ Z*d))
	return F

# define a new class pertm_gen: a generator for the PERT distribution

def best_fit(DF,col_value,col_var):

	df = DF.copy()
	def std(DF):
		df = DF.copy()
		df[col_var] = df[col_value].pct_change()
		std = df["return"].std()
		return std

	def CAGR(DF):
		df = DF.copy()
		n =len(df)/4
		df["return"] = df["Dividend"].pct_change()
		df["cum_return"] = (1+df["return"]).cumprod()
		CAGR = ((df["cum_return"].iloc[-1])**(1/n)) -1
		return CAGR
	#df_ratio = pd.read_sql("SELECT * FROM RatioData WHERE ID_STOCK = {}".format(id_instrument),acc_engine).sort_values("Data_Appended").iloc[-1]
	last_price = df[col_value].iloc[-1]

	std_data = df[[col_value]].pct_change().dropna()
	std_data = std_data[std_data[col_value] != 0]
	std_data = std_data[col_value]
	mean = sum(std_data)/len(std_data)
	std = moment(std_data,2)**0.5
	std_data = (std_data-mean) / std
	coeff = fit_fleishman_from_standardised_data(std_data)
	sim = (generate_fleishman(-coeff[1],*coeff,N=10000))*std+mean
	kurt = df[[col_value]].pct_change().dropna().kurt()
	skew =df[[col_value]].pct_change().dropna().skew()
	kurt = kurt.values[0]
	skew = skew.values[0]
	
	def rand_skew_norm(fAlpha, fLocation, fScale):
		sigma = fAlpha / np.sqrt(1.0 + fAlpha**2) 
		afRN = np.random.randn(2)
		u0 = afRN[0]
		v = afRN[1]
		u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 
		if u0 >= 0:
			return u1*fScale + fLocation 
		return (-u1)*fScale + fLocation 


	def randn_skew(N, skew=0.0):
		return [rand_skew_norm(skew, 0, 1) for x in range(N)]
	
	#xmin = df[[col_value]].pct_change().dropna().min()
	#xmax = df[[col_value]].pct_change().dropna().max()
	#X = np.linspace(min(std_data), max(std_data))
	#skew_dist = skewnorm.pdf(X, *skewnorm.fit(std_data))

	f = Fitter(std_data,distributions= get_common_distributions())
	f.fit()
	best_fit = pd.DataFrame(f.summary())
	best_fit.reset_index(inplace = True)
	best_fit = best_fit.iloc[0]["index"]
	return best_fit
#monte_carlo_simulation_robust(1)

def best_dist_(id_instrument,beta_model,valuation_model):

	if beta_model == "Market":
		df_regression,df_returns = get_betas(id_instrument,"Market",52,1)
	elif beta_model == "Carhart":
		df_regression,df_returns = get_betas(id_instrument,"Market 3 Moment",52,1)
	elif beta_model =="Market 3 Moment":
		df_regression,df_returns = get_betas(id_instrument,"Carhart",52,1)
		
	else:
		pass

	mean_beta = df_regression["Beta_Market"].mean()
	std_beta = df_regression["Beta_Market"].std()
	if valuation_model == "DDM":
		div_rate = df_ratio["div_rate"]
		div = last_price*div_rate
		g_rate = growth_rate/100

		## LATER FIND A SOLUTION TO CHANGE THIS 
		div_data_omega = pd.read_csv("omega_healthcare_div.csv")
		div_data_omega["Ex-Dividend Date"] = pd.to_datetime(div_data_omega["Ex-Dividend Date"])
		div_data_omega = div_data_omega.sort_values("Ex-Dividend Date")
		growth_rate_div = CAGR(div_data_omega)



		div_mean = div_data_omega["Dividend"].pct_change().mean()
		div_std = std(div_data_omega)
		div_dist = np.random.normal(loc = div_mean,scale = div_std, size = iterations)
		beta_dist =np.random.normal(loc = mean_beta,scale = std_beta, size = iterations)
		output_distributions = []
		market_return = market_return/100
		ke_dist = beta_dist*market_return
		for i in range(iterations):
			new_div_g = div*(1+div_dist[i])
			new_div = new_div_g*div
			new_ke = ke_dist[i]
			discount_factor = new_ke #- g_rate
			value_per_share = new_div/discount_factor
			output_distributions.append(value_per_share)

		perc_25 = np.percentile(output_distributions,25)
		mean_value = np.mean(output_distributions)
		plt.hist(output_distributions, bins = iterations, density = False)
		plt.axvline(last_price, color='k', linestyle='dashed', linewidth=1)
		plt.xlim([-20, 200])
		plt.show()

	else:
		pass


