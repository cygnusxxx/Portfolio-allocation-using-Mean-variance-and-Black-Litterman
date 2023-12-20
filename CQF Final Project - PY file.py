#!/usr/bin/env python
# coding: utf-8

# Fetching raw price data of all asset classes from spreadsheet

# In[1]:


# importing libraries

import numpy as np
import array as arr
import pandas as pd
import scipy as sci
import scipy.optimize as sco
import seaborn as sn
import datetime
import sklearn
from numpy import *
from numpy.linalg import multi_dot
from scipy.linalg import eigh, cholesky
from scipy.stats import norm

# Visualizaiton
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Reading raw data from spreadsheet

file =(r'C:\Users\Gunjan\Desktop\Price data.xls')
rawData = pd.read_excel(file)
rawData

df = pd.DataFrame(rawData)
df[["Date1", "Date2", "Date3", "Date4", "Date5", "Date6", "Date7", "Date8", "Date9"]] = df[["Date1", "Date2", "Date3", "Date4", "Date5", "Date6", "Date7", "Date8", "Date9"]].apply(pd.to_datetime)

df1 = df.iloc[:,0:2]
filt1 = (df1['Date1'] >= pd.to_datetime('2016-08-10'))
df1 = df1.loc[filt1].set_index("Date1")
df1["Treasuries"] = df1.pct_change().fillna(0)

df2 = df.iloc[:,2:4]
filt2 = (df2['Date2'] >= pd.to_datetime('2016-08-10'))
df2 = df2.loc[filt2].set_index("Date2")
df2["Commodities"] = df2.pct_change().fillna(0)

df3 = df.iloc[:,4:6]
filt3 = (df3['Date3'] >= pd.to_datetime('2016-08-10'))
df3 = df3.loc[filt3].set_index("Date3")
df3["Equities"] = df3.pct_change().fillna(0)

df4 = df.iloc[:,6:8]
filt4 = (df4['Date4'] >= pd.to_datetime('2016-08-10'))
df4 = df4.loc[filt4].set_index("Date4")
df4["TIPS"] = df4.pct_change().fillna(0)

df5 = df.iloc[:,8:10]
filt5 = (df5['Date5'] >= pd.to_datetime('2016-08-10'))
df5 = df5.loc[filt5].dropna().set_index("Date5")
df5["Gold"] = df5.pct_change().fillna(0)

df6 = df.iloc[:,10:12]
filt6 = (df6['Date6'] >= pd.to_datetime('2016-08-10'))
df6 = df6.loc[filt6].dropna().set_index("Date6")
df6["US-IG"] = df6.pct_change().fillna(0)

df7 = df.iloc[:,12:14]
filt7 = (df7['Date7'] >= pd.to_datetime('2016-08-10'))
df7 = df7.loc[filt7].dropna().set_index("Date7")
df7["US-HY"] = df7.pct_change().fillna(0)

df8 = df.iloc[:,14:16]
filt8 = (df8['Date8'] >= pd.to_datetime('2016-08-10'))
df8 = df8.loc[filt8].dropna().set_index("Date8")
df8["US-Agency MBS"] = df8.pct_change().fillna(0)

df9 = df.iloc[:,16:18]
filt9 = (df9['Date9'] >= pd.to_datetime('2016-08-10'))
df9 = df9.loc[filt9].dropna().set_index("Date9")
df9["VIX"] = df9.pct_change().fillna(0)

df10 = df.iloc[:,18:20]
filt10 = (df10['Date10'] >= pd.to_datetime('2016-08-10'))
df10 = df10.loc[filt10].dropna().set_index("Date10")
rfr = (df10.mean())/100

df_final = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8], axis=1)
df_final

df_final_indices = df_final.iloc[:,[0,2,4,6,8,10,12,14]]
df_final_returns = df_final.iloc[:,[1,3,5,7,9,11,13,15]]
df_final_returns

annual_returns = (df_final_returns.mean() * 252)
annual_returns


# In[4]:


fig = plt.figure(figsize=(15,5))
ax =plt.axes()
ax.bar(annual_returns.index, annual_returns*100, color='royalblue', alpha=0.8)
ax.set_title('Annualized Returns (in %)');


# In[5]:


asset_names = ['Treasuries', 'Commodities', 'Equities', 'TIPS', 'Gold', 'US-IG', 'US-HY', 'US-Agency MBS']

df_rf = pd.DataFrame(rfr[0], columns=['Excess Returns'], index= asset_names)
df_annual_returns = pd.DataFrame(annual_returns, columns=['Excess Returns'], index= asset_names)
df_excess_returns = df_annual_returns.subtract(df_rf)
df_excess_returns

fig = plt.figure(figsize=(15,5))
ax =plt.axes()
ax.bar(df_excess_returns.index, df_excess_returns['Excess Returns']*100, color='royalblue', alpha=0.8)
ax.set_title('Annualized Excess Returns (in %)');


# In[6]:


df_excess_returns


# In[7]:


vols = df_final_returns.std()
vols
annual_vols = vols*sqrt(252)
annual_vols
an_vols = np.array(annual_vols)[:, newaxis]

df_annual_vols = pd.DataFrame(an_vols, columns=['Annual vols'], index= asset_names)
df_annual_vols


# In[8]:


fig = plt.figure(figsize=(15,5))
ax = plt.axes()
ax.bar(annual_vols.index, annual_vols*100, color='red', alpha=0.8)
ax.set_title('Annualized Volatility (in %)');


# In[9]:


excess_ret = df_excess_returns.to_numpy(dtype ='float32')
an_vols = np.array(annual_vols)[:, newaxis]
sharpe_ratios = np.divide(excess_ret, an_vols)
df_sharpe_ratios = pd.DataFrame(sharpe_ratios, columns=['Sharpe Ratio'], index= asset_names)
df_sharpe_ratios


# In[10]:


df_hist_stats = pd.concat([df_excess_returns,df_annual_vols,df_sharpe_ratios], axis=1)
df_hist_stats


# In[11]:


fig = plt.figure(figsize=(15,5))
ax = plt.axes()
ax.bar(asset_names, df_sharpe_ratios['Sharpe Ratio'], color='green', alpha=0.8)
ax.set_title('Sharpe ratios');


# Naive Sample Covariance

# In[12]:


cov = df_final_returns.cov() * 252

plt.figure(figsize = (15,5))
ax = sn.heatmap(cov, annot=True)


# In[13]:


corr = df_final_returns.corr()
df_corr = pd.DataFrame(corr, index= asset_names,
                  columns= asset_names)
plt.figure(figsize = (15,5))
ax = sn.heatmap(df_corr, annot=True)


# Eigen Values of Naive sample corrleation matrix

# In[14]:


eVal, eVec = np.linalg.eigh(corr)
indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
eVal, eVec = eVal[indices], eVec[:, indices]

eVal


# In[15]:


plt.figure(figsize = (15,5))
plt.bar(np.arange(len(eVal)),eVal)
plt.title('Sorted eigenvalues')
plt.grid()
plt.show()


# Marcenko-Pastur Denoising (Robust Covariance)

# In[16]:


def cov2corr(cov):
    std = np.sqrt(np.diag(cov)) 
    corr = cov/np.outer(std,std) 
    corr[corr<-1] = -1
    corr[corr>1] = 1
    return corr

def corr2cov(corr, std): 
    cov=corr*np.outer(std,std) 
    return cov


# In[17]:


nFacts = np.sum(eVal > 0.5)

def denoisedCorr(eVal, eVec, nFacts):
    eVal_ = eVal.copy() 
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0]-nFacts) 
    eVal_ = np.diag(eVal_)
    corr = np.dot(eVec,eVal_).dot(eVec.T) 
    corr = cov2corr(corr)
    return corr


# In[18]:


cor_d = denoisedCorr(eVal, eVec, nFacts)
df_cor = pd.DataFrame(cor_d, index= asset_names,
                  columns= asset_names)
plt.figure(figsize = (15,5))
ax = sn.heatmap(df_cor, annot=True)


# In[19]:


cov_d = corr2cov(cor_d, np.diag(cov)**.5)
df_cov = pd.DataFrame(cov_d, index= asset_names,
                  columns= asset_names)
plt.figure(figsize = (15,5))
ax = sn.heatmap(df_cov, annot=True)


# Eigen values after Marsenko-Pastur Denoising (Robust covariance)

# In[20]:


eVal_d, eVec_d = np.linalg.eigh(cor_d)
indices = eVal_d.argsort()[::-1]  # arguments for sorting eVal desc
eVal_d, eVec_d = eVal_d[indices], eVec_d[:, indices]

eVal_d


# In[21]:


plt.figure(figsize = (15,5))
plt.bar(np.arange(len(eVal_d)),eVal_d)
plt.title('Eigenvalues AFTER denoising')
plt.grid()
plt.show()


# Constrained Mean Variance Optimization using Naive covariance with market implied lamda = 12.1

# In[22]:


num_assets = len(asset_names)
weights = random.random(num_assets)[:, newaxis]
lamda = 12.1
excess_ret = df_excess_returns.to_numpy(dtype ='float32')
covar = cov

#function to be used for optimisation
def portfolio_stats_NaiveMV(weights):
    
    port_rets = np.dot(weights.T, excess_ret)    
    port_vols = sqrt(multi_dot([weights.T, (lamda/2)*covar, weights])) 
    
    return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

#function which will be optimized

def mean_variance_constrained(weights):
    return (portfolio_stats_NaiveMV(weights)[1]**2) - portfolio_stats_NaiveMV(weights)[0]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0.05, 0.3) for x in range(num_assets))
initial_wts = num_assets*[1/num_assets]

# Optimizing for naive mean variance
opt_mean_var_naive = sco.minimize(mean_variance_constrained, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Portfolio weights
list(zip(asset_names,np.around(opt_mean_var_naive['x']*100,2)))

df_a = pd.DataFrame(np.around(opt_mean_var_naive['x']*100,2), index = asset_names, columns =['Optimal Weights with Naive covariance'])
df_a


# In[23]:


#function to calculate final optimised portfolio stats
def portfolio_stat_(weights):
    
    port_rets = np.dot(weights.T, excess_ret)    
    port_vols = sqrt(multi_dot([weights.T, covar, weights])) 
    
    return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats, portfolio_stat_(opt_mean_var_naive['x'])))
portfolio_stat_(opt_mean_var_naive['x'])

df_b = pd.DataFrame(np.around(np.vstack(portfolio_stat_(opt_mean_var_naive['x'])),2), index = stats, columns =['Portfolio stats with Naive covariance'])
df_b


# Constrained Mean Variance Optimization using Robust covariance with market implied lamda = 12.1

# In[24]:


num_assets = len(asset_names)
weights = random.random(num_assets)[:, newaxis]
lamda = 12.1
excess_ret = df_excess_returns.to_numpy(dtype ='float32')
covar = cov_d

#function to be used for optimisation
def portfolio_stats_RobustMV(weights):
    
    port_rets = np.dot(weights.T, excess_ret)    
    port_vols = sqrt(multi_dot([weights.T, (lamda/2)*covar, weights])) 
    
    return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

#function which will be optimized

def mean_variance_constrained(weights):
    return (portfolio_stats_RobustMV(weights)[1]**2) - portfolio_stats_RobustMV(weights)[0]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0.05, 0.3) for x in range(num_assets))
initial_wts = num_assets*[1/num_assets]

# Optimizing for robust mean variance
opt_mean_var_robust = sco.minimize(mean_variance_constrained, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Portfolio weights
list(zip(asset_names,np.around(opt_mean_var_robust['x']*100,2)))

df_c = pd.DataFrame(np.around(opt_mean_var_robust['x']*100,2), index = asset_names, columns =['Optimal Weights with Robust covariance'])
df_c


# In[25]:


# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats, portfolio_stat_(opt_mean_var_robust['x'])))

df_d = pd.DataFrame(np.around(np.vstack(portfolio_stat_(opt_mean_var_robust['x'])),2), index = stats, columns =['Portfolio stats with Robust covariance'])
df_d


# In[26]:


df_naive_vs_robust = pd.concat([df_a,df_c], axis=1)
df_naive_vs_robust.index.name = 'Asset class'
df_naive_vs_robust

df_naive_vs_robust.plot(kind='bar', figsize = (15,4))
plt.title('Optimal Mean variance allocations using Naive vs Robust covariance')
plt.grid()
plt.show()


# In[27]:


df_naive_vs_robust_stats = pd.concat([df_b,df_d], axis=1)
df_naive_vs_robust_stats.index.name = 'Portfolio Stats'
df_naive_vs_robust_stats

df_naive_vs_robust_stats.plot(kind='bar', figsize = (15,4))
plt.title('Portfolio stats using Naive vs Robust covariance')
plt.grid()
plt.show()


# Black-Litterman Model (Using robust covariance and Unconstrained Mean Variance optimizer with market implied lamda = 12.1)

# In[28]:


num_assets = len(asset_names)
data = {'Index weights':[0.21, 0.11, 0.21, 0.21, 0.06, 0.06, 0.06, 0.08]}
df_eq_wts = pd.DataFrame(data, index = asset_names)
df_eq_wts


# In[29]:



Index_weights = [0.21, 0.11, 0.21, 0.21, 0.06, 0.06, 0.06, 0.08]
Eq_wts = arr.array('d', Index_weights)
excess_ret = df_excess_returns.to_numpy(dtype ='float32')
excess_ret_mkt = np.dot(excess_ret.T,Eq_wts)
covar = cov_d
Eq_wts = array(Eq_wts)[:,newaxis]
sigma_mkt = sqrt(multi_dot([Eq_wts.T, covar, Eq_wts]))
sharpe_ratio = excess_ret_mkt/sigma_mkt
tau = 0.025


def Black_Litterman(covar,sharpe_ratio,tau):

    #Index (Equilibrium) weights for asset classes
    
    Index_weights = [0.21, 0.11, 0.21, 0.21, 0.06, 0.06, 0.06, 0.08]
    Eq_wts = arr.array('d', Index_weights)
    Eq_wts = array(Eq_wts)[:,newaxis]

    #Robust covariance matrix for excess returns

    cov_d_inv = np.linalg.inv(covar)
    sigma_mkt = sqrt(multi_dot([Eq_wts.T, covar, Eq_wts]))
    
    #calculation of lambda (Risk aversion coefficient)

    lamda = (sharpe_ratio)/(sigma_mkt)

    #Reverse optimization

    Eq_risk_premium = lamda*np.dot(covar, Eq_wts)
    num_datapoints = len(df_final_returns.axes[0])
    #tau = 1/num_datapoints

    #covariance of prior distribution

    tau_cov_inv = np.linalg.inv(tau*covar)
    
    #Investor views: 
    # View 1) Treasuries will underperform equities by 5%;  
    # View 2) Gold will have an absolute excess return of 10%
    
    View1 = [-1, 0, 1, 0, 0, 0, 0, 0]
    View2 = [0, 0, 0, 0, 1, 0, 0, 0]
    Qvec = [0.05, 0.10]

    P_vec_list = [View1, View2]
    P_vec = np.array(P_vec_list)

    Q_vec = arr.array('d', Qvec)
    Q_vec = array(Q_vec)[:,newaxis]

    #Matrix P.tau.covar.Ptranspose

    Matrix = multi_dot([P_vec, tau*covar, P_vec.T])

    Omega = np.diag(np.diag(Matrix))
    Omega_inv = np.linalg.inv(Omega)

    Matrix_A = tau_cov_inv + multi_dot([P_vec.T, Omega_inv, P_vec])
    Matrix_B = np.dot(tau_cov_inv, Eq_risk_premium) + multi_dot([P_vec.T, Omega_inv, Q_vec])

    #Expected return and covariance for posterior distribution

    ret_posterior = np.dot(np.linalg.inv(Matrix_A), Matrix_B)
    cov_posterior = covar + np.linalg.inv(tau_cov_inv + multi_dot([P_vec.T, Omega_inv, P_vec]))

    #Asset allocation with unconstrained optimization

    BL_weights =  np.dot((1/lamda)*cov_d_inv, ret_posterior)

    return np.array([lamda, sharpe_ratio, ret_posterior, cov_posterior, BL_weights]).flatten() 



# Portfolio weights
list(zip(asset_names,np.around(Black_Litterman(covar,sharpe_ratio,tau)[4]*100,2)))

df_e = pd.DataFrame(np.around(Black_Litterman(covar,sharpe_ratio,tau)[4]*100,2), index = asset_names, columns =['BL Unconstrained MV optimizer, λ=12.1'])
df_e


# In[30]:


weights = Black_Litterman(covar,sharpe_ratio,tau)[4]

#function to calculate final optimised portfolio stats
def portfolio_stats_(weights):
    
    port_rets = np.dot(weights.T, Black_Litterman(covar,sharpe_ratio,tau)[2])    
    port_vols = sqrt(multi_dot([weights.T, Black_Litterman(covar,sharpe_ratio,tau)[3], weights])) 
    
    return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

portfolio_stats_(weights)

df_m = pd.DataFrame(np.around(portfolio_stats_(weights),4), index = stats, columns =['BL Unconstrained MV optmizer, λ=12.1'])
df_m


# Black-Litterman Model (Using robust covariance and Constrained Mean Variance optimizer with market implied lamda = 12.1)

# In[31]:


weights = random.random(num_assets)[:, newaxis]
covar = cov_d

#function to be used for optimisation
def portfolio_stats_MV(weights):
    
    port_rets = np.dot(weights.T, Black_Litterman(covar,sharpe_ratio,tau)[2])    
    port_vols = sqrt(multi_dot([weights.T, (Black_Litterman(covar,sharpe_ratio,tau)[0]/2)*Black_Litterman(covar,sharpe_ratio,tau)[3], weights])) 
    
    return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

#function which will be optimized

def mean_variance_constrained(weights):
    return (portfolio_stats_MV(weights)[1]**2) - portfolio_stats_MV(weights)[0]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0.05, 0.3) for x in range(num_assets))
initial_wts = num_assets*[1/num_assets]

# Optimizing for constrained mean variance 
opt_mean_var = sco.minimize(mean_variance_constrained, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Portfolio weights
list(zip(asset_names,np.around(opt_mean_var['x']*100,2)))

df_f = pd.DataFrame(np.around(opt_mean_var['x']*100,2), index = asset_names, columns =['BL Constrained MV optimizer, λ=12.1'])
df_f


# In[32]:


# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats, portfolio_stats_(opt_mean_var['x'])))

df_g = pd.DataFrame(np.around(np.vstack(portfolio_stats_(opt_mean_var['x'])),4), index = stats, columns =['BL Constrained MV optmizer, λ=12.1'])
df_g


# Black-Litterman Model (Using robust covariance and Constrained Min Variance optimizer with market implied lamda = 12.1)

# In[33]:


weights = random.random(num_assets)[:, newaxis]
covar = cov_d

# Minimize Variance
def min_variance(weights):
    return portfolio_stats_(weights)[1]**2

# Each asset boundary ranges from 0 to 1
tuple((0, 1) for x in range(num_assets))

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0.05, 0.3) for x in range(num_assets))
initial_wts = num_assets*[1/num_assets]

# Optimizing for constrained minimum variance
opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)


# Portfolio weights
list(zip(asset_names,np.around(opt_var['x']*100,2)))

df_h = pd.DataFrame(np.around(opt_var['x']*100,2), index = asset_names, columns =['BL Constrained Min Var optimizer'])
df_h


# In[34]:


# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats,portfolio_stats_(opt_var['x'])))

df_i = pd.DataFrame(np.around(np.vstack(portfolio_stats_(opt_var['x'])),2), index = stats, columns =['BL Constrained Min Var optmizer'])
df_i


# Black-Litterman Model (Using robust covariance and Constrained Max Sharpe optimizer with market implied lamda = 12.1)

# In[35]:


weights = random.random(num_assets)[:, newaxis]
covar = cov_d

# Maximizing sharpe ratio
def min_sharpe_ratio(weights):
    return -portfolio_stats_(weights)[2]

# Each asset boundary ranges from 0 to 1
tuple((0, 0.3) for x in range(num_assets))

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 2) for x in range(num_assets))
initial_wts = num_assets*[1/num_assets]

# Optimizing for maximum sharpe ratio
opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Portfolio weights
list(zip(asset_names,np.around(opt_sharpe['x']*100,2)))

df_j = pd.DataFrame(np.around(opt_sharpe['x']*100,2), index = asset_names, columns =['BL Constrained Max Sharpe optimizer'])
df_j


# In[36]:


# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats,portfolio_stats_(opt_sharpe['x'])))

df_k = pd.DataFrame(np.around(np.vstack(portfolio_stats_(opt_sharpe['x'])),4), index = stats, columns =['BL Constrained Max Sharpe optmizer'])
df_k


# In[37]:


df_optimizers = pd.concat([df_e,df_f,df_h,df_j], axis=1)
df_optimizers.name = 'Asset class'
df_optimizers

df_optimizers.plot(kind='bar', figsize = (15,5))
plt.title('Black Litterman Robust variance Allocations using Mean variance vs Min Var vs Max Sharpe Constrained Optimizers')
plt.legend(loc= 'upper left')
plt.grid()
plt.show()


# In[38]:


df_optimizers_stats = pd.concat([df_g,df_i,df_k], axis=1)
df_optimizers_stats = df_optimizers_stats.T
df_optimizers_stats

df_optimizers_stats.plot(y=['Returns','Volatility'], kind='bar', figsize = (15,5))
ax= df_optimizers_stats['Sharpe Ratio'].plot(secondary_y=True, color='green', marker='o')
ax.set_ylabel('Sharpe Ratio')
plt.title('Portfolio stats for Black Litterman Robust variance Allocations using Mean variance vs Min Var vs Max Sharpe Constrained Optimizers')
plt.grid()
plt.show()


# Black-Litterman Model (Using robust covariance and Unconstrained Mean Variance optimizer with lamda = 1.4)

# In[39]:


covar = cov_d
sharpe_ratio = 0.1
tau = 0.025

Black_Litterman(covar,sharpe_ratio,tau)

weights = Black_Litterman(covar,sharpe_ratio,tau)[4]

df_n = pd.DataFrame(np.around(portfolio_stats_(weights),4), index = stats, columns =['BL Unconstrained MV optmizer, λ=1.4'])
df_n


# In[40]:


df_l = pd.DataFrame(np.around(Black_Litterman(covar,sharpe_ratio,tau)[4]*100,2), index = asset_names, columns =['BL Unconstrained MV optimizer, λ=1.4'])
df_l


# Black-Litterman Model (Using robust covariance and Constrained Mean Variance optimizer with lamda = 1.4)

# In[41]:


weights = random.random(num_assets)[:, newaxis]

portfolio_stats_MV(weights)

# Mean-Variance constrained

mean_variance_constrained(weights)
    
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0.05, 0.3) for x in range(num_assets))
initial_wts = num_assets*[1/num_assets]

# Optimizing for maximum sharpe ratio
opt_mean_var = sco.minimize(mean_variance_constrained, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Portfolio weights
list(zip(asset_names,np.around(opt_mean_var['x']*100,2)))

df_q = pd.DataFrame(np.around(opt_mean_var['x']*100,2), index = asset_names, columns =['BL Constrained MV optimizer, λ=1.4'])
df_q


# In[42]:


# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats, portfolio_stats_(opt_mean_var['x'])))

df_r = pd.DataFrame(np.around(np.vstack(portfolio_stats_(opt_mean_var['x'])),4), index = stats, columns =['BL Constrained MV optmizer, λ=1.4'])
df_r


# Black-Litterman Model (Using robust covariance and Unconstrained Mean Variance optimizer with lamda = 7)

# In[43]:


covar = cov_d
sharpe_ratio = 0.5
tau = 0.025

Black_Litterman(covar,sharpe_ratio,tau)

weights = Black_Litterman(covar,sharpe_ratio,tau)[4]


df_o = pd.DataFrame(np.around(portfolio_stats_(weights),4), index = stats, columns =['BL Unconstrained MV optmizer, λ=7'])
df_o


# In[44]:


df_p = pd.DataFrame(np.around(Black_Litterman(covar,sharpe_ratio,tau)[4]*100,2), index = asset_names, columns =['BL Unconstrained MV optimizer, λ=7'])
df_p


# Black-Litterman Model (Using robust covariance and Constrained Mean Variance optimizer with lamda = 7)

# In[45]:


weights = random.random(num_assets)[:, newaxis]

portfolio_stats_MV(weights)

# Mean-Variance constrained

mean_variance_constrained(weights)
    
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0.05, 0.3) for x in range(num_assets))
initial_wts = num_assets*[1/num_assets]

# Optimizing for maximum sharpe ratio
opt_mean_var = sco.minimize(mean_variance_constrained, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Portfolio weights
list(zip(asset_names,np.around(opt_mean_var['x']*100,2)))

df_s = pd.DataFrame(np.around(opt_mean_var['x']*100,2), index = asset_names, columns =['BL Constrained MV optimizer, λ=7'])
df_s


# In[46]:


# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats, portfolio_stats_(opt_mean_var['x'])))

df_t = pd.DataFrame(np.around(np.vstack(portfolio_stats_(opt_mean_var['x'])),4), index = stats, columns =['BL Constrained MV optmizer, λ=7'])
df_t


# In[47]:


df_lambdas_unconst = pd.concat([df_l,df_p,df_e], axis=1)
df_lambdas_unconst

df_lambdas_unconst.plot(kind='bar', figsize = (15,5))
plt.title('Black Litterman Robust variance Allocations using Mean variance unconstrained Optimizers for different lambdas')
plt.legend(loc= 'upper left')
plt.grid()
plt.show()


# In[48]:


df_lambdas_unconst_stats = pd.concat([df_n,df_o,df_m], axis=1)
df_lambdas_unconst_stats = df_lambdas_unconst_stats.T
df_lambdas_unconst_stats

df_lambdas_unconst_stats.plot(y=['Returns','Volatility'], kind='bar', figsize = (15,5))
ax= df_lambdas_unconst_stats['Sharpe Ratio'].plot(secondary_y=True, color='green', marker='o')
ax.set_ylabel('Sharpe Ratio')
plt.title('Portfolio stats for Black Litterman Robust variance Allocations using Mean variance unconstrained Optimizers for different lambdas')
plt.grid()
plt.show()


# In[49]:


df_lambdas_const = pd.concat([df_q,df_s,df_f], axis=1)
df_lambdas_const

df_lambdas_const.plot(kind='bar', figsize = (15,5))
plt.title('Black Litterman Robust variance Allocations using Mean variance constrained Optimizers for different lambdas')
plt.legend(loc= 'upper left')
plt.grid()
plt.show()


# In[50]:


df_lambdas_const_stats = pd.concat([df_r,df_t,df_g], axis=1)
df_lambdas_const_stats = df_lambdas_const_stats.T
df_lambdas_const_stats

df_lambdas_const_stats.plot(y=['Returns','Volatility'], kind='bar', figsize = (15,5))
ax= df_lambdas_const_stats['Sharpe Ratio'].plot(secondary_y=True, color='green', marker='o')
ax.set_ylabel('Sharpe Ratio')
plt.title('Portfolio stats for Black Litterman Robust variance Allocations using Mean variance constrained Optimizers for different lambdas')
plt.grid()
plt.show()


# End of the code
