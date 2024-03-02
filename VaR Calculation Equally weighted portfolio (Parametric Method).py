#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm


# In[2]:


#lets set the time (Duration of the data)
years = 15
endDate  = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365*years)


# In[3]:


tickers = ['SPY','BND','GLD','QQQ','VTI']
   


# In[4]:


#creating the dataframe of the data we want for calculatine returns
adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker,start = startDate, end = endDate)
    adj_close_df[ticker] = data['Adj Close']
print(adj_close_df)


# In[5]:


# daily log returns(These are net (r) not R)
# log return are easier for calculation
# this shift 1 is used to calculate the the (number /number-shifted 1 above) it

log_returns = np.log(adj_close_df/adj_close_df.shift(1))
log_returns = log_returns.dropna()

log_returns


# In[6]:


# now we will create equally weighted portfolio

portfolio_value = 1000000
weights = np.array([1/len(tickers)]*len(tickers))
print(weights)


# In[7]:


#now we will calculate the historical portfolio return

historical_returns = (log_returns * weights).sum(axis = 1)
print(historical_returns)


# In[8]:


days = 5

historical_x_days_returns = historical_returns.rolling(window = days).sum()


# In[9]:


#create covariance matrix for all the assets
#252 trading days in a year

cov_matrix = log_returns.cov()*252 


# In[10]:


#portfolio standard deviation
portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)


# In[11]:


# confidence levels to visualise
confidence_level = [0.90,0.95,0.99]


# In[12]:


VaRs = []
for c1 in confidence_level:
    VaR = portfolio_value * ((norm.ppf(1-c1)*portfolio_std_dev *np.sqrt(days/252))-historical_returns.mean()*days) 
    VaRs.append(VaR)


# In[13]:


VaRs


# In[15]:


#convert the data into dollar values 
historical_x_days_returns_dollar = historical_x_days_returns * portfolio_value

#histogram
plt.hist(historical_x_days_returns_dollar , bins = 50 , density = True , alpha = 0.5,label = f'{days}-Day Returns')

#add vertical line to see the Var
for c1,VaR in zip(confidence_level, VaRs):
    plt.axvline(x= VaR , linestyle = '--', color = 'r', label = 'VaR at {}% Confidence'.format(int(c1*100)))

plt.xlabel(f'{days}-Day Portfolio Returns ($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of portfolio{days}-Day Returns and Parametric VaR Estimates')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




