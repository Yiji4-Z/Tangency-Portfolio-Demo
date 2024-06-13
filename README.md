# Portfolio Management Project: Tangency Portfolio &  Capital Market Line

---

&nbsp;&nbsp; This Project implemented Scipy Optimization Method to construct a tangency portfolio with stocks from Tesla, Apple, J.P. Morgan, Amazon, and Costco. It also includes the corresponding Capital Market Line with assumption of 5% One-Year Treasury Bill Rate. Efficiency of this portfolio construction is measured by Sharpe Ratio of the next year portfolio performance. The following is a brief outline of the project:
<font color = "darkgrey">
* Data Processing
* Exploratory Data Analysis
* Risk & Return Relationship of Portfolio
* Maximal Sharpe Ratio & Minimal Risk
* Efficiency Frontier & Capital Market Line
* Verification for Efficiency of Portfolio Construction </font>

#### Reference

---
[1] : Brealey, Myers, and Allen Chapters 7 and 8.1 

[2] : Algorithmic Portfolio Optimization in Python: <https://kevinvecmanis.io/finance/optimization/2019/04/02/Algorithmic-Portfolio-Optimization.html>

[3] : Kaggle Portfolio Management:
<https://www.kaggle.com/code/harshalnikose/portfolio-management> 



### Data Processing<font color = "darkgrey"> 

---

* Imported various Python Packages 
* Extracted Stock Data from Yahoo Finance</font>


```python
#Packages Imported
import numpy as np
import pandas as pd
from yahoofinancials import YahooFinancials
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plot
import scipy.optimize as spyop


```


```python
#Load Data
TSLA = yf.download("TSLA", start="2012-05-20", end="2024-06-10",group_by="ticker") # Stock of Google
AAPL = yf.download("AAPL", start="2012-05-20", end="2024-06-10",group_by="ticker") # Stock of Apple
JPM = yf.download("JPM", start="2012-05-20", end="2024-06-10",group_by="ticker") # Stock of Facebook
AMZN = yf.download("AMZN", start="2012-05-20", end="2024-06-10",group_by="ticker") # Stock of Amazon
COST = yf.download("COST", start="2012-05-20", end="2024-06-10",group_by="ticker") # Stock of Microsoft
STK_list = [TSLA, AAPL, JPM, AMZN, COST]
print(TSLA.shape, AAPL.shape, JPM.shape, AMZN.shape,COST.shape,)


```

    (3032, 6) (3032, 6) (3032, 6) (3032, 6) (3032, 6)


    



```python
#Combine Data into one Dataframe
STK_df = pd.concat([company['Close'] for company in STK_list], axis=1)
STK_df.columns = ['TSLA', 'AAPL', 'JPM', 'AMZN', 'COST']
STK_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TSLA</th>
      <th>AAPL</th>
      <th>JPM</th>
      <th>AMZN</th>
      <th>COST</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-05-21</th>
      <td>1.918000</td>
      <td>20.045713</td>
      <td>32.509998</td>
      <td>10.9055</td>
      <td>83.730003</td>
    </tr>
    <tr>
      <th>2012-05-22</th>
      <td>2.053333</td>
      <td>19.891787</td>
      <td>34.009998</td>
      <td>10.7665</td>
      <td>83.379997</td>
    </tr>
    <tr>
      <th>2012-05-23</th>
      <td>2.068000</td>
      <td>20.377144</td>
      <td>34.259998</td>
      <td>10.8640</td>
      <td>83.309998</td>
    </tr>
    <tr>
      <th>2012-05-24</th>
      <td>2.018667</td>
      <td>20.190001</td>
      <td>33.970001</td>
      <td>10.7620</td>
      <td>84.480003</td>
    </tr>
    <tr>
      <th>2012-05-25</th>
      <td>1.987333</td>
      <td>20.081785</td>
      <td>33.500000</td>
      <td>10.6445</td>
      <td>84.480003</td>
    </tr>
  </tbody>
</table>
</div>



### Exploratory Data Analysis

---

<font color = "darkgrey">
&nbsp;&nbsp;
Based on the graph of the Stock Close Price v.s. Date, the years before 2016 performed quite different from the following years, so only data after 2016 will be included as references.<br />

&nbsp;&nbsp;Specifically data from 2016-06 to 2023-06 will be used to measure the yearly average return and yearly risk (fluctuations) of the stock to approach the optimal weights of each stock for the Tangency Portfolio.<br />

&nbsp;&nbsp;Daily Return is measured in log returns, which emphasizes negative values, and such property is consistant with our unwantedness of loss.
</font>


```python
STK_df.plot()
```




    <Axes: xlabel='Date'>




    
![png](portfolio%20management%20project_files/portfolio%20management%20project_5_1.png)
    



```python
#Measure the Daily return based on the close price of each stock
STK_df_raw = np.log(STK_df/STK_df.shift(1)).dropna()
STK_df_return = STK_df_raw[((STK_df_raw.index >= '2016-06-01') & (STK_df_raw.index < '2023-06-01')) ]
```

#### Related Mathematics Formula

---
* **$i$** is an abstract index for each stock, such that $i \in \{1,2,3,4,5 \} $, representing the 5 stocks in the portfolio
* Average return For Each Stock:
$$
    \mu_i = E[R_i]
$$

* Standard Deviation (Risk)  For Each Stock:
$$

    \sigma_i = \sqrt{E[(R_i - \mu_i)^2]} 
$$
* Covariance & Correlation between Stocks: 
$$
    Cov[R_i, R_j] = E[(R_i-\mu_i)(R_j-\mu_j)]
$$
$$
    Corr_{ij} (\rho_{ij}) = \frac{Cov[R_i, R_j]}{\sigma_i \sigma_j}
$$

* **Overall Standard Deviation (Risk) for the Portfolio with weights ($\omega_i$):**
$$
    \sqrt{\sum_{i=1}^n \omega_i^2 \sigma_i^2 + \sum_{i\neq j} \omega_i \omega_j \sigma_i \sigma_j \rho_{ij}}
    
$$


```python
#calculate mean, standard deviation and covariance
R_mu = STK_df_return.mean() * 252
plot.bar(R_mu.index, R_mu, color = 'mediumaquamarine')
plot.title("Bar Chart for Average Return")
```




    Text(0.5, 1.0, 'Bar Chart for Average Return')




    
![png](portfolio%20management%20project_files/portfolio%20management%20project_8_1.png)
    



```python
R_std = STK_df_return.std() * np.sqrt(252)
plot.bar(R_std.index, R_std, color = 'cadetblue')
plot.title("Bar Chart for Magnitude of Risk")
```




    Text(0.5, 1.0, 'Bar Chart for Magnitude of Risk')




    
![png](portfolio%20management%20project_files/portfolio%20management%20project_9_1.png)
    



```python
R_corr = STK_df_return.corr()
R_corr.style.background_gradient(cmap='GnBu')

```




<style type="text/css">
#T_9e4a1_row0_col0, #T_9e4a1_row1_col1, #T_9e4a1_row2_col2, #T_9e4a1_row3_col3, #T_9e4a1_row4_col4 {
  background-color: #084081;
  color: #f1f1f1;
}
#T_9e4a1_row0_col1 {
  background-color: #f3fbed;
  color: #000000;
}
#T_9e4a1_row0_col2, #T_9e4a1_row0_col4, #T_9e4a1_row2_col0, #T_9e4a1_row2_col1, #T_9e4a1_row2_col3 {
  background-color: #f7fcf0;
  color: #000000;
}
#T_9e4a1_row0_col3 {
  background-color: #daf0d4;
  color: #000000;
}
#T_9e4a1_row1_col0 {
  background-color: #c9eac4;
  color: #000000;
}
#T_9e4a1_row1_col2, #T_9e4a1_row3_col4 {
  background-color: #cdebc6;
  color: #000000;
}
#T_9e4a1_row1_col3 {
  background-color: #87d1c0;
  color: #000000;
}
#T_9e4a1_row1_col4 {
  background-color: #bae4bd;
  color: #000000;
}
#T_9e4a1_row2_col4, #T_9e4a1_row3_col2 {
  background-color: #eef9e8;
  color: #000000;
}
#T_9e4a1_row3_col0 {
  background-color: #d3eecd;
  color: #000000;
}
#T_9e4a1_row3_col1 {
  background-color: #b5e2bb;
  color: #000000;
}
#T_9e4a1_row4_col0 {
  background-color: #f2faeb;
  color: #000000;
}
#T_9e4a1_row4_col1 {
  background-color: #e1f3dc;
  color: #000000;
}
#T_9e4a1_row4_col2 {
  background-color: #e9f7e3;
  color: #000000;
}
#T_9e4a1_row4_col3 {
  background-color: #d0ecc9;
  color: #000000;
}
</style>
<table id="T_9e4a1">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9e4a1_level0_col0" class="col_heading level0 col0" >TSLA</th>
      <th id="T_9e4a1_level0_col1" class="col_heading level0 col1" >AAPL</th>
      <th id="T_9e4a1_level0_col2" class="col_heading level0 col2" >JPM</th>
      <th id="T_9e4a1_level0_col3" class="col_heading level0 col3" >AMZN</th>
      <th id="T_9e4a1_level0_col4" class="col_heading level0 col4" >COST</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9e4a1_level0_row0" class="row_heading level0 row0" >TSLA</th>
      <td id="T_9e4a1_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_9e4a1_row0_col1" class="data row0 col1" >0.458474</td>
      <td id="T_9e4a1_row0_col2" class="data row0 col2" >0.267061</td>
      <td id="T_9e4a1_row0_col3" class="data row0 col3" >0.416746</td>
      <td id="T_9e4a1_row0_col4" class="data row0 col4" >0.288892</td>
    </tr>
    <tr>
      <th id="T_9e4a1_level0_row1" class="row_heading level0 row1" >AAPL</th>
      <td id="T_9e4a1_row1_col0" class="data row1 col0" >0.458474</td>
      <td id="T_9e4a1_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_9e4a1_row1_col2" class="data row1 col2" >0.445725</td>
      <td id="T_9e4a1_row1_col3" class="data row1 col3" >0.628734</td>
      <td id="T_9e4a1_row1_col4" class="data row1 col4" >0.513189</td>
    </tr>
    <tr>
      <th id="T_9e4a1_level0_row2" class="row_heading level0 row2" >JPM</th>
      <td id="T_9e4a1_row2_col0" class="data row2 col0" >0.267061</td>
      <td id="T_9e4a1_row2_col1" class="data row2 col1" >0.445725</td>
      <td id="T_9e4a1_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_9e4a1_row2_col3" class="data row2 col3" >0.301937</td>
      <td id="T_9e4a1_row2_col4" class="data row2 col4" >0.322676</td>
    </tr>
    <tr>
      <th id="T_9e4a1_level0_row3" class="row_heading level0 row3" >AMZN</th>
      <td id="T_9e4a1_row3_col0" class="data row3 col0" >0.416746</td>
      <td id="T_9e4a1_row3_col1" class="data row3 col1" >0.628734</td>
      <td id="T_9e4a1_row3_col2" class="data row3 col2" >0.301937</td>
      <td id="T_9e4a1_row3_col3" class="data row3 col3" >1.000000</td>
      <td id="T_9e4a1_row3_col4" class="data row3 col4" >0.462207</td>
    </tr>
    <tr>
      <th id="T_9e4a1_level0_row4" class="row_heading level0 row4" >COST</th>
      <td id="T_9e4a1_row4_col0" class="data row4 col0" >0.288892</td>
      <td id="T_9e4a1_row4_col1" class="data row4 col1" >0.513189</td>
      <td id="T_9e4a1_row4_col2" class="data row4 col2" >0.322676</td>
      <td id="T_9e4a1_row4_col3" class="data row4 col3" >0.462207</td>
      <td id="T_9e4a1_row4_col4" class="data row4 col4" >1.000000</td>
    </tr>
  </tbody>
</table>




### Risk & Return Relationship of Portfolio

---

<font color = "darkgrey">

* Define function for measuring the overall risk of the Portfolio
* Simulate 10,000 groups of weights for stocks in the Portfolio, and plot the resulted return and risk
* Color the return & risk pair with its Sharpe Ratio
</font>


```python
# Assume 1 yr treasury bill rate is 5%
TRSRY = 0.05
```


```python
def Std_Portfolio(weights, std_ls, corr_mx):
    var_port = 0
    for i in range(len(std_ls)):
        var_port = var_port + weights[i]**2 * std_ls.iloc[i]**2 
        for j in range(i+1, len(std_ls)):
            var_port = var_port+ 2*weights[i]*weights[j]*std_ls.iloc[i]*std_ls.iloc[j]*corr_mx.iloc[i,j]
    std_port = np.sqrt(var_port)
    return std_port
    
```


```python
# sample 10,000 combination of stocks and plot them
def Stat_Portfolio(weights, mu_ls, std_ls, corr_mx, T_rate ):
    pm = sum([weights[l] * R_mu.iloc[l] for l in range(len(mu_ls))]) 
    pstd = Std_Portfolio(weights,std_ls,corr_mx) 
    psharpe = (pm-T_rate)/pstd
    return [pm, pstd, psharpe]

iter = 10000
pm = []
pstd = []
psharpe = []

for k in range(iter):   
    weights_p = np.random.dirichlet(np.ones(len(STK_df_return.columns)),size = 1)[0]
    port_stat = Stat_Portfolio(weights_p, R_mu, R_std, R_corr, TRSRY)
    pm.append(port_stat[0])
    pstd.append(port_stat[1])
    psharpe.append(port_stat[2])



```


```python
plot.scatter(pstd, pm, c = psharpe, s=5)
plot.colorbar(label="Sharpe Ratio with risk-free rate as 5%")
```




    <matplotlib.colorbar.Colorbar at 0x1690d4bb0>




    
![png](portfolio%20management%20project_files/portfolio%20management%20project_15_1.png)
    


### Maximal Sharpe Ratio & Minimal Risk

---

<font color="darkgrey">

* Define functions, constraints, bounds, and initial condition for applying Scipy Optimization
* Plot the Point with Minimal Risk & the Point with Minimal Sharpe Ratio
</font>


```python
def Min_Risk(weights):
    return Stat_Portfolio(weights, R_mu, R_std, R_corr, TRSRY)[1]

def Max_Sharpe(weights):
    return -Stat_Portfolio(weights, R_mu, R_std, R_corr, TRSRY)[2]

def M_Return(weights):
     return Stat_Portfolio(weights, R_mu, R_std, R_corr, TRSRY)[0]
```


```python
Cons = ({'type': 'eq','fun': lambda w: np.sum(w) - 1})
Bnds = (((0,1),) * len(STK_df.columns))
Init = [1/len(STK_df.columns)] * len(STK_df.columns)
print(Bnds)
print(Init)
```

    ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
    [0.2, 0.2, 0.2, 0.2, 0.2]



```python
#Find the weights combination with minimal risk
Min_Risk_W = spyop.minimize(Min_Risk,Init, 
                            method = 'SLSQP', 
                            bounds = Bnds, 
                            constraints = Cons )['x']
Min_Risk_W
```




    array([2.90921971e-18, 4.03388539e-02, 2.88045489e-01, 9.17036033e-02,
           5.79912054e-01])




```python
Min_Risk_result = Stat_Portfolio(Min_Risk_W, R_mu, R_std, R_corr, TRSRY)
Min_Risk_pm = Min_Risk_result[0]
Min_Risk_pstd = Min_Risk_result[1]
```


```python
#Find the weights combination with maximal Sharpe Ratio
Max_Sharpe_W = spyop.minimize(Max_Sharpe,
                              Init, method = 'SLSQP', 
                              bounds = Bnds, 
                              constraints = Cons )['x']
Max_Sharpe_W
```




    array([1.24682991e-01, 5.92662788e-01, 1.23599048e-17, 0.00000000e+00,
           2.82654221e-01])




```python
Max_Sharpe_result = Stat_Portfolio(Max_Sharpe_W, R_mu, R_std, R_corr, TRSRY)
Max_Sharpe_pm = Max_Sharpe_result[0]
Max_Sharpe_pstd = Max_Sharpe_result[1]
Max_Sharpe_psharpe = Max_Sharpe_result[2]
```


```python
fig, ax = plot.subplots()
grph = ax.scatter(pstd, pm, c = psharpe, s=5)
ax.set_xlabel("Risk")
ax.set_ylabel("Return")
plot.colorbar(grph, label="Sharpe Ratio with risk-free rate as 5%")

ax.scatter(Min_Risk_pstd, Min_Risk_pm,marker = '*', c='b',edgecolor = 'white', s = 300)
ax.scatter(Max_Sharpe_pstd, Max_Sharpe_pm,marker = '*', c='orange',edgecolor = 'white', s = 300)
ax.annotate(f'Minimal Risk is {Min_Risk_pstd.round(4)}', (Min_Risk_pstd, Min_Risk_pm), weight = "bold")
ax.annotate(f'Max Sharpe Ratio is {Max_Sharpe_psharpe.round(4)}', (Max_Sharpe_pstd, Max_Sharpe_pm), weight = "bold")
```




    Text(0.26138452018512626, 0.2627463731862672, 'Max Sharpe Ratio is 0.8139')




    
![png](portfolio%20management%20project_files/portfolio%20management%20project_23_1.png)
    


### Efficiency Frontier & Capital Market Line

---

<font color = "darkgrey">

* Find Points with Maximal Sharpe Ratio for given returns as illustration for the Efficiency Frontier
* Use the Point with Maximal Sharpe Ratio and the Point for the risk-free return to construct the Capital Market Line
</font>


```python
target_m= np.linspace(Min_Risk_pm,np.max(pm),50)

EF_m = []
EF_std = []
for target in target_m:
    Cons_EF = ({'type': 'eq','fun': lambda w: np.sum(w) - 1},
                {'type': 'eq','fun': lambda w: M_Return(w) - target})
    w = spyop.minimize(Max_Sharpe,Init, method = 'SLSQP', bounds = Bnds, constraints = Cons_EF )['x']
    r = Stat_Portfolio(w, R_mu, R_std, R_corr, TRSRY)
    EF_m.append(r[0])
    EF_std.append(r[1])

```


```python
fig, ax = plot.subplots()
grph = ax.scatter(pstd, pm, c = psharpe, s=5)
ax.set_xlabel("Risk")
ax.set_ylabel("Return")
plot.colorbar(grph, label="Sharpe Ratio with risk-free rate as 5%")

ax.scatter(EF_std, EF_m, marker = 'P', c='orange', edgecolors='w')
ax.scatter(Min_Risk_pstd, Min_Risk_pm,marker = '*', c='b',edgecolor = 'white', s = 300)
ax.scatter(Max_Sharpe_pstd, Max_Sharpe_pm,marker = '*', c='orange',edgecolor = 'white', s = 300)
ax.annotate(f'Minimal Risk is {Min_Risk_pstd.round(4)}', (Min_Risk_pstd, Min_Risk_pm), weight = "bold")
ax.annotate(f'Max Sharpe Ratio is {Max_Sharpe_psharpe.round(4)}', (Max_Sharpe_pstd, Max_Sharpe_pm), weight = "bold")


```




    Text(0.26138452018512626, 0.2627463731862672, 'Max Sharpe Ratio is 0.8139')




    
![png](portfolio%20management%20project_files/portfolio%20management%20project_26_1.png)
    



```python
fig, ax = plot.subplots()
grph = ax.scatter(pstd, pm, c = psharpe, s=5)
ax.set_xlabel("Risk")
ax.set_ylabel("Return")
plot.colorbar(grph, label="Sharpe Ratio with risk-free rate as 5%")

x1, y1 = [0, Max_Sharpe_pstd], [TRSRY, Max_Sharpe_pm]
ax.plot(x1, y1, marker='.')

ax.scatter(EF_std, EF_m, marker = 'P', c='orange', edgecolors='w')
ax.scatter(Min_Risk_pstd, Min_Risk_pm,marker = '*', c='b',edgecolor = 'white', s = 300)
ax.scatter(Max_Sharpe_pstd, Max_Sharpe_pm,marker = '*', c='orange',edgecolor = 'white', s = 300)
ax.annotate(f'Minimal Risk is {Min_Risk_pstd.round(4)}', (Min_Risk_pstd, Min_Risk_pm), weight = "bold")
ax.annotate(f'Max Sharpe Ratio is {Max_Sharpe_psharpe.round(4)}', (Max_Sharpe_pstd, Max_Sharpe_pm), weight = "bold")


```




    Text(0.26138452018512626, 0.2627463731862672, 'Max Sharpe Ratio is 0.8139')




    
![png](portfolio%20management%20project_files/portfolio%20management%20project_27_1.png)
    


### Verification for Efficiency of Portfolio Construction 

---

<font color = 'darkgrey'>
&nbsp;&nbsp; Based on the statistics of the portfolio with equal weights and the portfolio with optimal weights, it is clear that the optimal weights produced a higher return average of 26.27%, a slightly lower risk 17.36%, and a sharpe ratio **1.225** which is 24%  higher than the portfolio with equal weights.

</font>



```python
STK_df_test = STK_df_raw[(STK_df_raw.index >= '2023-06-01') & (STK_df_raw.index <= '2024-06-20')]
tst_pm = STK_df_test.mean() * 252
tst_pstd = STK_df_test.std() * np.sqrt(252)
tst_corr = STK_df_test.corr()
```


```python
#With Equal Weights
Eql_w = Stat_Portfolio(Init, tst_pm, tst_pstd, tst_corr, TRSRY)
Opt_w = Stat_Portfolio(Max_Sharpe_W, tst_pm, tst_pstd, tst_corr, TRSRY)
print(list(zip(['eql_return', 'eql_risk', 'eql_sharpe ratio'], list(Eql_w))))
print(list(zip(['opt_return', 'opt_risk', 'opt_sharpe ratio'], list(Opt_w))))
print(f'Reletive Increase in Sharpe Ratio:  {(Opt_w[2] - Eql_w[2])/Eql_w[2]}')

```

    [('eql_return', 0.22167100938989304), ('eql_risk', 0.17414875511282993), ('eql_sharpe ratio', 0.9857722455648243)]
    [('opt_return', 0.2627463731862672), ('opt_risk', 0.17364380234407006), ('opt_sharpe ratio', 1.225188404736246)]
    Reletive Increase in Sharpe Ratio:  0.24287167776187676



```python
iter_test = 10000
pm = []
pstd = []
psharpe = []

for k in range(iter_test):   
    weights_p = np.random.dirichlet(np.ones(len(STK_df_test.columns)),size = 1)[0]
    port_stat_tst = Stat_Portfolio(weights_p, tst_pm, tst_pstd, tst_corr, TRSRY)
    pm.append(port_stat_tst[0])
    pstd.append(port_stat_tst[1])
    psharpe.append(port_stat_tst[2])

```


```python
fig, ax = plot.subplots()
grph = ax.scatter(pstd, pm, c = psharpe, s=5)
ax.set_xlabel("Risk")
ax.set_ylabel("Return")
plot.colorbar(grph, label="Sharpe Ratio with risk-free rate as 5%")

ax.scatter(Opt_w[1],Opt_w[0],marker = '*', c='b',edgecolor = 'white', s = 300)
```




    <matplotlib.collections.PathCollection at 0x16973d790>




    
![png](portfolio%20management%20project_files/portfolio%20management%20project_32_1.png)
    


### Conclusion

---

&nbsp;&nbsp; This project successfully constructs a tangency portfolio with 5 stocks, and shows the optimalism of the weights of the tangency portfolio compared with other weights. The graph indicates that the optimal weights generated based on previous few years stock data still have a optimal risk & return performance among 10,000 weights combinations which clearly lies on the Efficiency Frontier with a high Sharpe Ratio.


