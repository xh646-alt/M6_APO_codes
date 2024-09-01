function x=meanvar(y,ll,caseType)

% Function to perform mean-variance portfolio optimization on a single dataset
%
% Inputs:
%   y  - raw data matrix of N random variables
%   ll - scalar constant value (risk aversion coefficient)
%   caseType  - Case selection for covariance calculation:
%               caseType = 1: Use standard covariance calculation (cov)
%               caseType = 2: Use shrinkage covariance calculation (cov1Para)
%
% Output:
%   x  - vector of optimized portfolio weights for the input dataset


N=size(y,2);

mean1=mean(y);

% Determine covariance matrix based on the selected case
if caseType==1
    covariance1=cov(y);         % Standard covariance calculation
elseif caseType==2
    covariance1=cov1Para(y);    % Parameterized covariance calculation
else
    error('Invalid caseType. Use 1 for cov or 2 for cov1Para.');
end

x=(1/N)*ones(N,1);
xo=(1/N)*ones(N,1);

LB=zeros(N,1);
UB=ones(N,1);

A=[];
B=[];

Aeq=ones(1,N);
Beq=1;

options=optimset('Algorithm','interior-point','TolFun',1e-05,'TolCon',1e-05,'MaxFunEvals',1000000000);

x=fmincon(@(x) -[mean1*x-0.5*ll*x'*covariance1*x],xo,A,B,Aeq,Beq,LB,UB,[],options);

end