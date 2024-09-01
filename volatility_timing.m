function x = volatility_timing(y,caseType)

% Function to perform Volatility Timing strategy on a single dataset
%
% Inputs:
%   y - raw data matrix of N random variables
%   caseType  - Case selection for covariance calculation:
%               caseType = 1: Use standard covariance calculation (cov)
%               caseType = 2: Use shrinkage covariance calculation (cov1Para)
%
% Output:
%   x - vector of optimized portfolio weights for the input dataset

% Determine covariance matrix based on the selected case
if caseType==1
    covariance=cov(y);         % Standard covariance calculation
elseif caseType==2
    covariance=cov1Para(y);    % Parameterized covariance calculation
else
    error('Invalid caseType. Use 1 for cov or 2 for cov1Para.');
end

inv_variances = 1./diag(covariance);

x = inv_variances/sum(inv_variances);

end