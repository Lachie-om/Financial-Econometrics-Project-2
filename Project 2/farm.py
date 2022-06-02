import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as web
import plotly.graph_objects as go

import statsmodels
# we will use `smf` and `sm` to constract and estimate same regressions but using different ways
import statsmodels.formula.api as smf  
import statsmodels.api as sm
import statsmodels.stats.api as sms
from patsy import dmatrices
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas_datareader import data
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
import statsmodels.tsa.api as smt
import pickle
import warnings
warnings.filterwarnings('ignore')

sns.set(color_codes=True)
sns.set_style('darkgrid')
plt.rc("figure", figsize=(16, 6))
plt.rc("savefig", dpi=500)
plt.rc("font",family="sans-serif")
plt.rc("font",size=14)

from arch.univariate import arch_model, ARX, ARCH, GARCH, StudentsT

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

import  pylab as pl

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
import numpy as np

from scipy.cluster.vq import kmeans,vq
from math import sqrt

from sklearn.cluster import KMeans

import pandas as pd
import pandas_datareader as dr

from matplotlib import pyplot as plt

# When editing a module, and not wanting to restatrt kernel every time use:
# import importlib
# importlib.reload(bc)
# import utsbootcamp as bc


def PageBreakPDF():
	# from IPython.display import display, Math, Latex
	# Usage: bc.PageBreakPDF()
	# Adds a page break in PDF output when saving Jupyter Notebook to PDF via LaTeX
	display(Latex(r"\newpage"))

def my_function():
    print('Hello you.')
	
def my_function2(name):
    print(f"Hello {name}, is it me you're looking for?")

def my_function3(name):
    print(f"Hello {name.capitalize()}, is it me you're looking for?")

def my_function4(name='alex'):
    if isinstance(name,str):
        print(f"Hello {name.capitalize()}, is it me you're looking for?")
    else:
        print('Inputs must be strings')	

def price2ret(x,keepfirstrow=False):
	ret = x.pct_change()
	if keepfirstrow:
		ret.fillna(0, inplace=True)
	else:
		ret.drop([ret.index[0]], inplace=True)
	return ret
	
def price2cret(x):
	ret = x.pct_change()
	ret.fillna(0, inplace=True)
	cret=((1 + ret).cumprod() - 1)
	return cret



#############################################
# Econometrics
#############################################

def regplot(df,formula,xName,yName):
	reg = smf.ols(formula, data=df).fit()
	print(reg.summary())
	x=df[xName]
	y=df[yName]
	yhat=reg.fittedvalues
	fig, ax = plt.subplots(figsize=(10,8))
	ax.plot(x, y, 'o', label="Raw data")
	ax.plot(x, yhat, 'r--.', label="OLS estimate")
	ax.legend(loc='best');
	return reg

def JBtest(resid,a=0.05):
    	# Residuals as input (reg.resid) and significance (dafault=0.05) 
    	test = sms.jarque_bera(reg.resid)
    	JBpvalue=test[1]
    	print(f'Jarque-Bera test:')
    	if JBpvalue<=a:
        	print(f'\tp-value is {JBpvalue:.03f}\n\tReject the null hypothesis that residuals are normally distributed. \n\tResiduals are NOT normally distributed.')
    	else:
        	print(f'\tp-value is {JBpvalue:.03f}\n\tFail to reject the null hypothesis that residuals are normally distributed. \n\tResiduals ARE normally distributed. ')
    	return JBpvalue

def BPtest(reg,a=0.05):
    	# Regression model as input (reg.resid) and significance (dafault=0.05) 
    	test = sms.het_breuschpagan(result.resid, result.model.exog)
    	BPpvalue=test[1]
    	print(f'Breusch-Pagan test:')
    	if BPpvalue<=a:
        	print(f'\tp-value is {BPpvalue:.03f}\n\tReject the null hypothesis of homoskedasticity. \n\tThe variance of the errors from a regression IS DEPENDENT on the values of the independent variables.')
    	else:
        	print(f'\tp-value is {BPpvalue:.03f}\n\tFail to reject the null hypothesis of homoskedasticity. \n\tThe variance of the errors from a regression does not depend on the values of the independent variables. ')
    	return BPpvalue


def VIF(df,formula):
	y, X = dmatrices(formula, data=df, return_type="dataframe")
	vif = pd.DataFrame()
	vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
	vif["Variable"] = X.columns
	return vif

def SimulateXY(b0=1,b1=2,n=1000,muX=0,sdX=1,errRatio=0.5):
	# Simulate data
	# n=1000           	# number of observations
	# muX=0 		# mean of X
	# sdX=1			# sd of X	
	# b0=1        		# define desired intercept for the line
	# b1=2        		# define desired slope of the line
	# errRatio=0.5 		# residual error relative to volatility of X variable

	# Simulate x data:
	x=np.random.normal(loc=muX,scale=sdX,size=(n,1))

	# Simulate errors. Errors must be with zero mean, but you can make standard deviation more or less than standard deviation of x (try!)
	err=np.random.normal(loc=0,scale=sdX*errRatio,size=(n,1))

	# Calculate y data:
	y = b0 + b1*x + err    # observed data (with error)
	y_true = b0 + b1*x     # true data

	df=pd.DataFrame(data=np.hstack((x,y)), columns=['x','y'])                   # Option 1
	# df = pd.DataFrame(np.concatenate([x,y], axis=1), columns= ['x','y'])      # Option 2
	
	return df, y_true

























class HTMLTableParser:
       
        def parse_url(self, url):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'lxml')
            return [(table['id'],self.parse_html_table(table))\
                    for table in soup.find_all('table')]  
    
        def parse_html_table(self, table):
            n_columns = 0
            n_rows=0
            column_names = []
    
            # Find number of rows and columns
            # we also find the column titles if we can
            for row in table.find_all('tr'):
                
                # Determine the number of rows in the table
                td_tags = row.find_all('td')
                if len(td_tags) > 0:
                    n_rows+=1
                    if n_columns == 0:
                        # Set the number of columns for our table
                        n_columns = len(td_tags)
                        
                # Handle column names if we find them
                th_tags = row.find_all('th') 
                if len(th_tags) > 0 and len(column_names) == 0:
                    for th in th_tags:
                        column_names.append(th.get_text())
    
            # Safeguard on Column Titles
            if len(column_names) > 0 and len(column_names) != n_columns:
                raise Exception("Column titles do not match the number of columns")
    
            columns = column_names if len(column_names) > 0 else range(0,n_columns)
            df = pd.DataFrame(columns = columns,
                              index= range(0,n_rows))
            row_marker = 0
            for row in table.find_all('tr'):
                column_marker = 0
                columns = row.find_all('td')
                for column in columns:
                    df.iat[row_marker,column_marker] = column.get_text()
                    column_marker += 1
                if len(columns) > 0:
                    row_marker += 1
                    
            # Convert to float if possible
            for col in df:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass
            
            return df
        
        
        ##### Project 1 #####
def read_data(ticker_list,
              start=dt.datetime(2015,1,1),
              end=dt.datetime(2022,3,1)):
    ticker=pd.DataFrame()
    
    for tick in ticker_list:
        prices=data.DataReader(tick,'yahoo',start, end)
        closing_prices=prices['Adj Close']
        ticker[tick]=closing_prices
        
    return ticker



def ADFtest(x,a=0.05,modeltype='c',autolagcriterion='BIC'):
    # x                : data
    # a                : significance level set at 95% C.I.
    # modeltype        : model type 
    # autolagcriterion : optimal lag selection method
    
    test=adfuller(x,regression=modeltype,autolag=autolagcriterion)
    print(f'Augmented Dickey-Fuller unit root test results:')
    print(f'\tTest statistics:\t\t {test[0]:9.4f}.')
    print(f'\tTest p-value:\t\t\t {test[1]:9.4f}.')
    print(f'\tNumber of lags selected:\t {test[2]:9.4f}.')
    if test[1]<=a:
        print('\tOUTCOME: Reject the null hypothesis. Series do not contain unit root.')
    else: 
        print('\tOUTCOME: Fail to reject the null hypothesis. Series appear to contain unit root.')
        

def KPSStest(x,a=0.05,modeltype='c',autolagcriterion='auto'):
    test=kpss(x,regression=modeltype,nlags=autolagcriterion)
    print(f'Kwiatkowski-Phillips-Schmidt-Shin test for stationarity results:')
    print(f'\tTest statistics:\t\t {test[0]:9.4f}.')
    print(f'\tTest p-value:\t\t\t {test[1]:9.4f}.')
    print(f'\tNumber of lags selected:\t {test[2]:9.4f}.')
    if test[1]<=a:
        print('\tOUTCOME: Reject the null hypothesis. Series is non-stationary.')
    else: 
        print('\tOUTCOME: Fail to reject the null hypothesis. Series appear to be stationary.')
        

def get_log_returns(data):
    # Get arithmetic returns
    arithmetic_returns = data.pct_change()
    # Transform to log returns
    arithmetic_returns = 1+arithmetic_returns
    returns_array = np.log(arithmetic_returns, out=np.zeros_like(arithmetic_returns), where=(arithmetic_returns != 0))
    return pd.DataFrame(returns_array, index=data.index, columns=data.columns).fillna(0) 


def plot_TS(e,f,g,nlags=40,a=0.05):
    
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18,17))
    ax[0,0].set_title("^DJI")
    ax[0,0].plot(e, linewidth=0.5, alpha=1)
    ax[0,1].set_title("TSLA")
    ax[0,1].plot(f, linewidth=0.5, alpha=1)
    ax[0,2].set_title("FCX")
    ax[0,2].plot(g, linewidth=0.5, alpha=1)
    sm.graphics.tsa.plot_acf(e, 
                         lags=nlags, alpha=a, 
                         title="Autocorrelation (ACF) of returns",
                         ax=ax[1,0])
    sm.graphics.tsa.plot_acf(f, 
                         lags=nlags, alpha=a, 
                         title="Autocorrelation (ACF) of returns",
                         ax=ax[1,1])
    sm.graphics.tsa.plot_acf(g, 
                         lags=nlags, alpha=a, 
                         title="Autocorrelation (ACF) of returns",
                         ax=ax[1,2])
    sm.graphics.tsa.plot_pacf(e, 
                         lags=nlags, alpha=a, 
                         title="Partial Autocorrelation (PACF) of returns", method='ywm',
                         ax=ax[2,0])
    sm.graphics.tsa.plot_pacf(f, 
                         lags=nlags, alpha=a, 
                         title="Partial Autocorrelation (PACF) of returns", method='ywm',
                         ax=ax[2,1])
    sm.graphics.tsa.plot_pacf(g, 
                         lags=nlags, alpha=a, 
                         title="Partial Autocorrelation (PACF) of returns", method='ywm',
                         ax=ax[2,2])
    plt.show()
    
    
def ARMA_BIC(x, n=5):
    BICValues = np.zeros((n+1,n+1))
    for p in range (n+1):
        for q in range (n+1):
            arma_mod = sm.tsa.arima.ARIMA(x, order=(p,0,q), trend='c')
            BICValues[p,q]= arma_mod.fit().bic
    return BICValues


def ARMA_AIC(x, n=5):
    AICValues = np.zeros((n+1,n+1))
    for p in range (n+1):
        for q in range (n+1):
            arma_mod = sm.tsa.arima.ARIMA(x, order=(p,0,q), trend='c')
            AICValues[p,q]= arma_mod.fit().aic
    return AICValues


def ARCH_test(x):
    res=x-np.mean(x)
    LM,LMpvalue,F,Fpvalue=statsmodels.stats.diagnostic.het_arch(res,nlags=20)
    return LMpvalue


def heatmap(x,y,v,b):
    plt.figure(figsize=(16,12))
    ax = sns.heatmap(x.transpose(),annot=True, cmap=plt.cm.coolwarm, fmt='2g', vmin=v,vmax=b)
    plt.xlabel('P', fontsize=15)
    plt.ylabel('Q', fontsize=15)
    ax.set_title(y)
    return plt.show()

def GARCH(x,n=5):
    GARCHValues = np.zeros((n+1,n+1))
    for i in range (0,n+1):
        for j in range (0,n+1):
            try:    
                garch_mod = arch_model(x,p=i,o=0,q=j,vol='GARCH')
                GARCHValues[i,j]= garch_mod.fit().bic
            except ValueError:
                pass
            
    return GARCHValues


def T_GARCH(x,n=5):
    TGARCHValues = np.zeros((n+1,n+1))
    for i in range (0,n+1):
        for j in range (0,n+1):
            try:    
                garch_mod = arch_model(x,p=i,o=0,q=j,vol='GARCH',dist='t')
                TGARCHValues[i,j]= garch_mod.fit().bic
            except ValueError:
                pass
            
    return TGARCHValues


def plotting_3D(o,p,i):
    fig = go.Figure(data=[go.Surface(z=o.transpose())])
    fig.update_layout(title=p, autosize=False,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90),
                  scene = dict(
                    xaxis_title='P',
                    yaxis_title='Q',
                    zaxis_title=i))
    return fig.show()