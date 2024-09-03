## Introduction

This repository contains the scripts to reproduce the results in "Single-stage Portfolio Optimization with Automated Machine Learning for M6". The codes provided in this respository implement model selection and optimize various regression models using the Method of Moving Asymptotes (Svanberg, 1987). It updates the weights iteratively to minimize the objective function while tracking the objective values and constraint function values throughout the process.

## Implemetation

Upload the Data;

Run Main.m: By executing Main.m, the user will obtain:

1. I. The main results for M6 per sub-period in the "all_sub_metrics", II. The total (global) results for M6 in "all_metrics", III. The S&P results with no transaction costs (TCs) in "all", IV. The S&P results with TCs in "post_TC_metrics".
  
2. For getting the S&P50 and S&P100 results, one needs to change the code in line 21 of the Main.m to "load sp50.mat" and "load sp100.mat", respectively, and re-run it.
   
3. For changing the TC estimates, one needs to change the code in line 198 of the Main.m to "ptc=5;" and "ptc=10;" for alternative TC estimates, and re-run.


The folder also contains the following helper functions:

-	‘get_optimal_coeff.m’: a function that calculates the coefficient for lasso, ridge, elastic net regression 

-	‘lasso_mma’: a function that calculates the lasso regression coefficients using the Method of Moving Asymptotes (MMA), it updates the weights iteratively to minimize the objective function with L1 regularization, while tracking the objective values and constraint function values throughout the process;

-	‘ridge_mma’: a function that calculates the ridge regression coefficients using the MMA optimization method;

-	‘enet_mma’: a function that calculates the elastic net regression coefficients using the MMA optimization method;

-	‘subsolv.m’: a function provided by Krister Svanberg that solves the subproblem in the MMA, efficiently updating the optimization variables while ensuring feasibility with respect to constraints;

-	‘mmasub.m’: a function developed by professor Krister Svanberg that performs one MMA-iteration, aimed at solving the nonlinear programming problem by a primal-dual Newton method;

-	‘performance2.m’: a function that evaluates the portfolio performance metrics, including return, standard deviation, and information ratio;

-	‘volatility_timing.m’: a function that calculates portfolio weights for a volatility timing strategy based on a single dataset

-	‘minvar.m’: a function that determines portfolio weights for a minimum-variance strategy using a single dataset;

-	‘meanvar.m’: a function that computes portfolio weights for a mean-variance strategy based on a single dataset;

-	‘Bayes_Stein.m’: a function that calculates portfolio weights for Bayes-Stein strategy for a single dataset;

-	‘Black_Litterman.m’: a function that determines the portfolio weights according to the Black-Litterman model;

-	‘TC.m’: a function that computes realized portfolio returns and performance metrics net of proportional transactions costs;

-	‘cov1Para.m’: a function developed by Ledoit and Wolf, that performs covariance matrix shrinkage, which improves the estimation of the covariance matrix by combining the sample covariance matrix with a structured target matrix.

The code was executed using MATLAB R2021b. 


Data (Only for replication purposes)

Historical price data for the M6 assets (covering the period from January 31, 2020, to January 31, 2022) were sourced from Yahoo Finance and Wharton Research Data Services (WRDS). 

The file ‘filtered_returns.csv’ contains the return data for S&P 500 constituents, obtained from Wharton Research Data Services (WRDS) and is used for additional analysis.

The MAT files ‘sp25.mat’, ‘sp50.mat’, and ‘sp100.mat’ provide the sampling indices for various scenarios. By executing df_returns = data(:, sampled_indices) on line 23 of Main.m, the datasets corresponding to N=25, N=50, and N=100 cases can be recovered.

Due to API limitations, the dataset used in this manuscript cannot be publicly shared.


## Acknowledgement
We are grateful to Professor Krister Svanberg for kindly sharing the MMA code.

Svanberg, K. (1987). The method of moving asymptotes—a new method for structural optimization. International journal for numerical methods in engineering, 24(2), 359-373.

@article{svanberg1987method,   
  title={The method of moving asymptotes—a new method for structural optimization},   
  author={Svanberg, Krister},   
  journal={International journal for numerical methods in engineering},   
  volume={24},   
  number={2},   
  pages={359--373},   
  year={1987},   
  publisher={Wiley Online Library}   
} 
