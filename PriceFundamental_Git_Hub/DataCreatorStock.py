import pandas as pd
import sys
from sqlite3 import connect
import urllib
from sqlalchemy import create_engine
import pyodbc
import urllib

conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=YOUR DIRECTORY\StockPriceFundamentalData.accdb;')
cnn_url = f"access+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"
acc_engine = create_engine(cnn_url)



def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except OSError as err:
        print(f"Error: '{err}'")



def createDataBase():
    conn_str = pyodbc.connect(r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ= YOUR DIRECTORY\StockPriceFundamentalData.accdb;')


    SectorIndex = """
    CREATE TABLE SectorIndex(
    ID_SECTOR INT NOT NULL PRIMARY KEY,
    NAME_SECTOR VARCHAR(50)
    );
    """

    SectorIndustryMerge = """
    CREATE TABLE SectorIndustryMerge(
    ID_SECTOR INT ,
    ID_INDUSTRY  INT NOT NULL PRIMARY KEY,
    FOREIGN KEY (ID_SECTOR) REFERENCES SectorIndex(ID_SECTOR)
    );
    """

    IndustryIndex = """
    CREATE TABLE IndustryIndex(
    ID_INDUSTRY INT NOT NULL PRIMARY KEY,
    NAME_INDUSTRY VARCHAR(50),
    Industry_Benchmark VARCHAR(100),
    Benchmark_Type VARCHAR(50),
    Benchmark_Symbol VARCHAR(50)
    );
    """

    IndustryStockMerge = """
    CREATE TABLE IndustryStockMerge(
    ID_INDUSTRY INT ,
    ID_STOCK INT NOT NULL PRIMARY KEY,
    FOREIGN KEY (ID_INDUSTRY) REFERENCES IndustryIndex(ID_INDUSTRY)
    );
    """

    StockIndex = """
    CREATE TABLE StockIndex(
    ID_STOCK INT NOT NULL PRIMARY KEY,
    SYMBOL VARCHAR(50),
    STOCK_NAME VARCHAR(50),
    FOREIGN KEY (ID_STOCK) REFERENCES IndustryStockMerge(ID_STOCK)
    );
    """


    StockPriceData = """
    CREATE TABLE StockPriceData(
    ID_STOCK INT,
    Data DATETIME,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    FOREIGN KEY (ID_STOCK) REFERENCES StockIndex(ID_STOCK)
    );
    """

    BenchmarkPriceData = """
    CREATE TABLE BenchmarkPriceData(
    ID_BENCH INT,
    Type VARCHAR(50),
    Data DATETIME,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    FOREIGN KEY (ID_BENCH) REFERENCES IndustryIndex(ID_INDUSTRY)
    );
    """

    AnalystRecommendation = """
    CREATE TABLE AnalystRecommendation(
    ID_STOCK INT, 
    Data DATETIME,
    firm VARCHAR(50),
    previous_grade VARCHAR(50),
    latest_grade VARCHAR(50),
    action_update VARCHAR(50),
    FOREIGN KEY (ID_STOCK) REFERENCES StockIndex(ID_STOCK)
    );
    """

    InstitutionalHolders = """
    CREATE TABLE InstitutionalHolders(
    ID_STOCK INT, 
    holder VARCHAR(50),
    Date_reported DATETIME, 
    perc_held FLOAT,
    FOREIGN KEY (ID_STOCK) REFERENCES StockIndex(ID_STOCK)
    );
    """
  

    RatioData = """
    CREATE TABLE RatioData(
    ID_STOCK INT, 
    Data_Appended DATETIME,
    eps FLOAT,
    pe_ratio FLOAT,
    forward_pe_ratio FLOAT,
    peg_ratio FLOAT,
    price_to_book_ratio FLOAT,
    price_to_sales FLOAT,
    price_to_cfo FLOAT,
    price_to_fcf FLOAT,
    div_rate FLOAT,
    ev_to_ebitda FLOAT,
    debt_to_equity FLOAT,
    cash_to_assets FLOAT,
    roa FLOAT,
    roe FLOAT,
    gross_margin FLOAT,
    operating_margin FLOAT,
    ni_margin FLOAT,
    coverage_ratio FLOAT,
    coverage_ratio_prime FLOAT,
    short_ratio FLOAT,
    number_of_analyst INT,
    target_mean FLOAT,
    terget_median FLOAT,
    insider_holder_perc FLOAT,
    institutional_holder_perc FLOAT,
    esg_score FLOAT,
    esg_performance VARCHAR(50),
    book_value_per_share FLOAT,
    cfo FLOAT,
    fcf FLOAT,
    sales FLOAT,
    shares_outstanding FLOAT,
    FOREIGN KEY (ID_STOCK) REFERENCES StockIndex(ID_STOCK)
    );
    """

    EarningsEstimatesData = """
    CREATE TABLE EarningsEstimate(
    ID_STOCK INT, 
    earnings_date DATETIME,
    earnings_estimate FLOAT,
    earnings_low FLOAT,
    earnings_high FLOAT,
    rev_estimate FLOAT,
    rev_low FLOAT,
    rev_high FLOAT,
    FOREIGN KEY (ID_STOCK) REFERENCES StockIndex(ID_STOCK))
    ;
    """
    

    IndustryAverage = """
    CREATE TABLE IndustryAverage(
    ID_INDUSTRY INT,
    Data_Appended DATETIME,
    pe_ratio_peer FLOAT,
    forward_pe_ratio_peer FLOAT,
    peg_ratio_peer FLOAT,
    price_to_book_ratio_peer FLOAT,
    price_to_sales_peer FLOAT,
    price_to_cfo_peer FLOAT,
    price_to_fcf_peer FLOAT,
    div_rate_peer FLOAT,
    ev_to_ebitda_peer FLOAT,
    debt_to_equity_peer FLOAT,
    cash_to_assets_peer FLOAT,
    roa_peer FLOAT,
    roe_peer FLOAT,
    gross_margin_peer FLOAT,
    operating_margin_peer FLOAT,
    ni_margin_peer FLOAT,
    coverage_ratio_peer FLOAT,
    coverage_ratio_prime_peer FLOAT,
    short_ratio_peer FLOAT,
    esg_score FLOAT,

    FOREIGN KEY (ID_INDUSTRY) REFERENCES IndustryIndex(ID_INDUSTRY)
    );
    """
    Co_skewness_Kurtosis  = """
    CREATE TABLE CoSkewnessKurt(
    ID_STOCK INT,
    ID_BENCH INT,
    Data DATETIME,
    corr_bench FLOAT,
    coskew_bench FLOAT,
    quadratic_cos FLOAT,
    unconditional_cos FLOAT,
    FOREIGN KEY (ID_STOCK) REFERENCES StockIndex(ID_STOCK)
    )
    """
    
    execute_query(conn_str,SectorIndex)
    execute_query(conn_str,SectorIndustryMerge)
    execute_query(conn_str,IndustryIndex)
    execute_query(conn_str,IndustryStockMerge)
    execute_query(conn_str,BenchmarkPriceData)
    execute_query(conn_str,StockIndex)
    execute_query(conn_str,StockPriceData)
    execute_query(conn_str,AnalystRecommendation)
    execute_query(conn_str,InstitutionalHolders)
    execute_query(conn_str,RatioData)
    execute_query(conn_str,IndustryAverage)
    execute_query(conn_str,Co_skewness_Kurtosis)



   execute_query(conn_str,EarningsEstimatesData)


createDataBase()




## Populate the dict table
