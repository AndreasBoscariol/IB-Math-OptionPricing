import math
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

omxs30_tickers = ["ABB.ST", "ALFA.ST",
"ASSA-B.ST", "ATCO-A.ST", "ATCO-B.ST", "BOL.ST", 
"ELUX-B.ST", "ERIC-B.ST", "ESSITY-B.ST", "GETI-B.ST",
"HEXA-B.ST", "INVE-B.ST", "KINV-B.ST",
"NDA-SE.ST", "SAND.ST", "SINCH.ST", "SEB-A.ST", "SKA-B.ST", "SKF-B.ST", "SCA-B.ST",
"SHB-A.ST", "SWED-A.ST", "SWMA.ST", "TEL2-B.ST", "TELIA.ST", "VOLV-B.ST"]

stock_data = {}
volatility = {}
price_list_MC = []
price_list_BS = []
price_list_Actual = []
lsd_list_MC = []
lsd_list_BS = []
strike_prices = []

for ticker in omxs30_tickers:
	stock_data[ticker] = yf.download(ticker, start='2017-01-01', end='2020-01-01')

	#Calculate Variance
	data = stock_data[ticker]['Close']
	mean = data.mean()
	variance = data.apply(lambda x: (x - mean)**2)
	sum_of_squared_variance = variance.sum()
	volatility[ticker] = (sum_of_squared_variance/(len(data)-1))**(1/2)
	
	#Set Variables
	r = 0.016 #risk free rate
	sigma = volatility[ticker] #variance
	T = 0.08 #time to maturity
	S0 = stock_data[ticker]['Close'][-1] #spot price


	sigma = returns.std()
	gamma = (mean - 0.5 * sigma**2) / sigma**2

	#range of strike price +/- 20%
	for k in range(80, 120):
		K=k/100*S0 #strike price
		strike_prices.append(K)

		# Modified Black Scholes
		m_sigma=0.2 * (1 + gamma * (t/T)**2)
	    d1 = (np.log(S/K) + (r+sigma(t)**2/2) * (T-t))/(sigma(t) * np.sqrt(T-t))
	    d2 = d1 - sigma(t) * np.sqrt(T-t)
	    phid1 = norm.cdf(d1)
	    phid2 = norm.cdf(d2)
	    call_price_BS = S * phid1 - K * np.exp(-r*(T-t)) * phid2

		#Calculate Black-Scholes Call Option
		d1 = (np.log(S0/K) + (r + sigma**2/2) * T)/(sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)
		phid1 = norm.cdf(d1)
		phid2 = norm.cdf(d2)
		call_price_BS = S0 * phid1 - K * math.exp(-r * T) * phid2
		print(call_price_BS)

		#Calculate Monte-Carlo Call Option
		nSim=1000000
		Z = np.random.normal(loc=0, scale=1, size=nSim)
		WT = np.sqrt(T) * Z
		ST = S0*np.exp((r - 0.5*sigma**2)*T + sigma*WT)
		simulated_call_payoffs = np.exp(-r*T)*np.maximum(ST-K,0)
		call_price_MC = np.mean(simulated_call_payoffs)
		print(call_price_MC)

		#Get Actual Value Call Option
		stock_data[ticker] = yf.download(ticker, start='2020-01-01', end='2020-01-31')
		call_price_Actual = stock_data[ticker]['Close'][-1]
		print(call_price_Actual)

		#Save all Prices
		price_list_MC.append(call_price_MC)
		price_list_BS.append(call_price_BS)
		price_list_Actual.append(call_price_Actual)


		# Calculate logarithmic standard deviation
		log_call_price_BS = np.log(call_price_BS)
		log_call_price_MC = np.log(call_price_MC)
		log_call_price_Actual = np.log(call_price_Actual)
		log_stdBS = np.std([log_call_price_BS, log_call_price_Actual])
		log_stdMC = np.std([log_call_price_MC, log_call_price_Actual])
		lsd_list_BS.append(log_stdBS)
		lsd_list_MC.append(log_stdMC)

#Average Logarithmic Standard Deviation	
average_lsd_MC = sum(lsd_list_MC) / len(lsd_list_MC)
print("MC:",average_lsd_MC)
average_lsd_BS = sum(lsd_list_BS) / len(lsd_list_BS)
print("BS:",average_lsd_BS)

#Sort Lists
strike_prices.sort()
price_list_BS.sort()
price_list_MC.sort()
price_list_Actual.sort()

#Plot Line Graph of MC, BS and Actual Call Option Values vs Strike Price
plt.plot(strike_prices, price_list_BS, label='Black-Scholes')
plt.plot(strike_prices, price_list_MC, label='Monte Carlo')
plt.plot(strike_prices, price_list_Actual, label='Actual')
plt.legend()
plt.xlabel('Strike Price')
plt.ylabel('Call Option Value'
)plt.title('Strike Price vs Call Option Value')
plt.show()

#Plot Bar Graph of Logarithmic Standard Deviation
x = ['Monte Carlo', 'Black-Scholes']
y = [average_lsd_MC, average_lsd_BS]

plt.bar(x, y)
plt.xlabel('Method')
plt.ylabel('Logarithmic Standard Deviation')
plt.title('LSD Comparison')
plt.show()







'''
print("Stock | Volatility")
print("-----------------")
for ticker in omxs30_tickers:
	print(ticker + " | " + str(volatility[ticker]))

#Plot Histogram of Stock History of ABB
plt.hist(stock_data["ABB.ST"]['Close'], bins=50)
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.title('Histogram of Closing Prices for ABB')
plt.show()
	
'''	


'''
["ABB.ST", "ALFA.ST",
"ASSA-B.ST", "AZN.ST", "ATCO-A.ST", "ATCO-B.ST", "ALIV-SDB.ST","BOL.ST", 
"ELUX-B.ST", "ERIC-B.ST", "ESSITY-B.ST", "EVO.ST", "GETI-B.ST",
"HM-B.ST", "HEXA-B.ST", "INVE-B.ST", "KINV-B.ST",
"NDA-SE.ST", "SAND.ST", "SINCH.ST", "SEB-A.ST", "SKA-B.ST", "SKF-B.ST", "SCA-B.ST",
"SHB-A.ST", "SWED-A.ST", "SWMA.ST", "TEL2-B.ST", "TELIA.ST", "VOLV-B.ST"]
'''