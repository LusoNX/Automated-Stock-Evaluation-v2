import pandas as pd
import sys
from sqlite3 import connect
import urllib
from sqlalchemy import create_engine
import pyodbc
import urllib

conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=C:\Users\35196\OneDrive\Desktop\Investment Prospectus\DataBase\Funds\FundDataBase.accdb;')
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
            r'DBQ=C:\Users\35196\OneDrive\Desktop\Investment Prospectus\DataBase\Funds\FundDataBase.accdb;')


    FundIndex = """
    CREATE TABLE FundIndex(
    ID_INSTRUMENT INT NOT NULL PRIMARY KEY,
    ISIN VARCHAR(50),
    fund_name VARCHAR(100),
    fund_currency VARCHAR(50),
    fund_country VARCHAR(50),
    manager VARCHAR(50),
    MorningStar_Code VARCHAR(50)
    );
    """


    PriceData = """
    CREATE TABLE FundPriceData(
    ID_INSTRUMENT INT,
    Data DATETIME,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    FOREIGN KEY (ID_INSTRUMENT) REFERENCES FundIndex(ID_INSTRUMENT)
    );
    """

    FactorIndex = """
    CREATE TABLE FactorIndex(
    ID_FACTOR INT,
    country VARCHAR(50),
    name DATETIME,
    Variable_name VARCHAR(50),
    Variable_value FLOAT,
    t_test FLOAT, 
    R2 FLOAT,
    FOREIGN KEY (ID_INDEX) REFERENCES IndexData(ID_INDEX)
    );
    """
    FactorPriceData = """
    CREATE TABLE FactorPriceData(
    ID_FACTOR  INT,
    Data DATETIME,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    FOREIGN KEY (ID_FACTOR) REFERENCES FactorIndex(ID_FACTOR)
    );
    """

    RegressionResults = """
    CREATE TABLE RegressionResults(
    ID_INDEX INT,
    Model VARCHAR(50),
    Data DATETIME,
    Variable_name VARCHAR(50),
    Variable_value FLOAT,
    t_test FLOAT, 
    R2 FLOAT,
    FOREIGN KEY (ID_INDEX) REFERENCES IndexData(ID_INDEX)
    );
    """





    
    
    #execute_query(conn_str,FundIndex)
    #execute_query(conn_str,PriceData)
    #execute_query(conn_str,Co_skewness_Kurtosis)
    execute_query(conn_str,RegressionResults)



    #Populate the dict data

createDataBase()




## Populate the dict table
