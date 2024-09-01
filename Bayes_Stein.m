function x=Bayes_Stein(y,K,ll,caseType)

% Function to perform Bayes-Stein portfolio optimization on a single dataset
%
% Inputs:
%   y         - Raw data matrix of N random variables (returns data)
%   K         - Scalar value
%   ll        - Risk aversion coefficient
%   caseType  - Case selection for covariance calculation:
%               caseType = 1: Use standard covariance calculation (cov)
%               caseType = 2: Use shrinkage covariance calculation (cov1Para)
%
% Output:
%   x - Vector of optimized portfolio weights for the input dataset


% Calculate mean and covariance
N=size(y,2);

mean1=mean(y);

% Determine covariance matrix based on the selected case
if caseType==1
    covariance1_opt=cov(y);         % Standard covariance calculation
elseif caseType==2
    covariance1_opt=cov1Para(y);    % Parameterized covariance calculation
else
    error('Invalid caseType. Use 1 for cov or 2 for cov1Para.');
end

X1=(inv(covariance1_opt)*ones(N,1)/(ones(1,N)*inv(covariance1_opt)*ones(N,1)));

mean_min_1=X1'*mean1(1:N)';

gama=(N+2)/(N+2+K*((mean1(1:N)'-mean_min_1*ones(N,1))')*inv(covariance1_opt)*(mean1(1:N)'-mean_min_1*ones(N,1)));

meanBS1=((1-gama)*mean1(1:N)'+gama*mean_min_1*ones(N,1));

meanBS1=meanBS1';

lll=((mean1(1:N)'-mean_min_1*ones(N,1))')*inv(covariance1_opt)*(mean1(1:N)'-mean_min_1*ones(N,1));

if lll==0

    lambda=100000000000000000000000000000000000000;

else

    lambda=(N+2)/lll;

end

covarianceBS1=(1+1/(K+lambda))*covariance1_opt+(lambda/(K*K+K+K*lambda))*ones(N,1)*ones(1,N)/(ones(1,N)*inv(covariance1_opt)*ones(N,1));

meanBS=meanBS1;
covarianceBS=covarianceBS1;

x=(1/N)*ones(N,1);
xo=(1/N)*ones(N,1);

LB=zeros(N,1);
UB=ones(N,1);

A=[];
B=[];

Aeq=ones(1,N);
Beq=1;

options=optimset('Algorithm','interior-point','TolFun',1e-05,'TolCon',1e-05,'MaxFunEvals',1000000000);

x=fmincon(@(x) -[meanBS*x-0.5*ll*x'*covarianceBS*x],xo,A,B,Aeq,Beq,LB,UB,[],options);

end