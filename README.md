# Automated-Stock-Evaluation-v2
Automatization of the evaluation process


The following project is an extension from the previous “Automated-Stock-Evaluation”, 
with improvements and additions to the relevant information for the valuation of stocks. 
The improvements are more related with the logistical process of the tool. 
A database is created DataCreatorStock, to accommodate the necessary data for valuation. 
The file stock_final_data.py is responsible to update the database when necessary, for both price and fundamental data. 
The data sources used are “Yahoo Finance” , “Investing.com” and “Alpha Vantage”. 
Once the data is updated, the remaining tools are employed to evaluate and extract the relevant information and statistics of the stock. 

![goi_margin_OHI](https://user-images.githubusercontent.com/84282116/170784568-59ef967a-52d3-44e7-aceb-ba360ab1e36e.png)
![coverage_ratio_OHI](https://user-images.githubusercontent.com/84282116/170784581-1fb45fcf-da6b-47e8-a3ac-45e5eb558db9.png)

![cfo_margin_OHI](https://user-images.githubusercontent.com/84282116/170784603-c8fdb718-1f81-45ef-8f28-c1280bf4006b.png)

The before mentioned additions relate firstly with some historical fundamental ratios (mostly margins) to understand the financial condition of the company, 
and secondly, more importantly, the addition relate to different forms to capture the cost of capital of the company. 
I still resort to traditional valuation models such has DDM and DCF, but now, the cost of capital is captured in three distinct models: 
-	The 1st model is called a 3 Moment CAPM and attempts to capture the effect of the second order effects of the market returns. 
	  That is on how second order variation in returns (1st order variation in Betas) affect the cost of equity. 
    This metric can be used to capture the coskewnewss risk between the market and the stock has shown in Moreno et al. (2009). 
    
    
-	The 2nd model is called the Carhart 4 Factor and attempts to capture 3 additional risk premiums (besides the mkt risk premium) within the model, 
      namely the size, value and momentum factors. 
      
      ![beta_4_moment_OHI](https://user-images.githubusercontent.com/84282116/170784395-6474b4e5-e364-4333-bf17-1784f61ebb29.png)

      
      
-	Finally the 3rd model is called “Industry Fama and French” and the model is similar to Fama and French 3 factor model, 
		with the exception that market returns are substituted by industry returns, in order to capture the industry risk premium. 
  Bear in mind, that the models are not applicable to all stocks. 
  These tools only function as a guideline to understand the risk premiums that drive stock prices, but by no means all the premiums are suitable for all the stocks. That is why the calculated betas are based on rolling windows, which allows us to observe how not only exposure to risk factors differ over time, but also if they remain statistically significant.  In a nutshell, all stocks differ, meaning that the valuation metrics based on the different cost of capital are not necessarily all appropriate at the same time. 
  
    ![beta_industry_ff_OHI](https://user-images.githubusercontent.com/84282116/170784472-ded3f4de-037e-49cc-af7f-941f62a24809.png)

  

An additional regression is made to capture the relation that the stock price has with conditional market movements. 
By using a Quantile Regression (QR) I attempt to capture how the alphas and Betas for each stock behave in situation of opportunism (upper bound) 
and stress (lower bound).
 


The final document is produced when you run doc_stock.py, but this will only work if you have data in the database. Thus follow the order:
1.	Run the files DataCreatorFunds. 
2.	Populate the main two tables - StockIndex and IndustryIndex from Fund and make sure you populate IndustryStockMerge to define which stocks relates to which industry, in order to get the ratio data for the IndustryAverage.
3.	Run the stock_final_data.py to get all the data and relevant ratios 
4.	Run doc_stock.py to get the final report in a docx format. 


![valuation_ddm_3_moment_OHI](https://user-images.githubusercontent.com/84282116/170784504-89642db2-c7b6-4cfe-a1ff-81697321f7b5.png)

[OHI Summary.docx](https://github.com/LusoNX/Automated-Stock-Evaluation-v2/files/8789554/OHI.Summary.docx)


Enjoy. 
** Notice ** This is by no means an approach to value companies, only a structured script that employs some of the common valuations used in traditional finance. Even though the results are somewhat useful if you are evaluating mature companies or companies with a predictable stream of cash flows (such as REITS), the results lack meaning when applied to growth companies or companies with unpredictable cash flows. 



