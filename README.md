## Introduction

This repository contains the scripts to reproduce the results Single-stage Portfolio Optimization with Automated Machine Learning for M6. The codes provided in this respository implement model selection and optimize various regression models using the Method of Moving Asymptotes (Svanberg, 1987). It updates the weights iteratively to minimize the objective function while tracking the objective values and constraint function values throughout the process.

## Implemetation
To reproduce the main results in the paper, execute the script Main.m. The output variable 'all_sub_metrics' provides performance metrics of APO for the 12 submission periods, while 'all_metrics' yields the global performance metrics.

## Acknowledgement
We are grateful to Professor Krister Svanberg for kindly sharing the code.

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
