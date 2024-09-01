function x=Black_Litterman(y,ll,caseType)

% Function to perform Black-Litterman portfolio optimization on a single dataset
%
% Inputs:
%   y         - Raw data matrix of N random variables (returns data)
%   ll        - Risk aversion coefficient
%   caseType  - Case selection for covariance calculation:
%               caseType = 1: Use standard covariance calculation (cov)
%               caseType = 2: Use shrinkage covariance calculation (cov1Para)
%
% Output:
%   x - Vector of optimized portfolio weights for the input dataset


% Set Black-Litterman parameters
c=ll;        
N=size(y,2);

tau=0.1;             % Scalar indicating the reliability in the equilibrium expected returns
P=diag(ones(N,1));  % Identity matrix representing that views are directly on all assets

% Determine covariance matrix based on the selected case
if caseType==1
    covariance1=cov(y);         % Standard covariance calculation
elseif caseType==2
    covariance1=cov1Para(y);    % Parameterized covariance calculation
else
    error('Invalid caseType. Use 1 for cov or 2 for cov1Para.');
end

% Euiqlibrium allocation
allocation=((inv(covariance1)*ones(N,1))/(ones(1,N)*inv(covariance1)*ones(N,1)))';

PP=ll*covariance1*allocation';
W=(1/c)*P*covariance1*P';

% Calculate experts' views
mean1=mean(y);
Q=mean1';

% Posterior mean and covariance
meanBL1=(inv(inv(tau*covariance1)+P'*inv(W)*P))*(inv(tau*covariance1)*PP+(P'*inv(W)*Q));
covarianceBL1=covariance1+inv(inv(tau*covariance1)+P'*inv(W)*P);

meanBL=meanBL1';
covarianceBL=covarianceBL1;

x=(1/N)*ones(N,1);
xo=(1/N)*ones(N,1);

LB=zeros(N,1);
UB=ones(N,1);

A=[];
B=[];

Aeq=ones(1,N);
Beq=1;

options=optimset('Algorithm','interior-point','TolFun',1e-05,'TolCon',1e-05,'MaxFunEvals',1000000000);

x=fmincon(@(x) -[meanBL*x-0.5*ll*x'*covarianceBL*x],xo,A,B,Aeq,Beq,LB,UB,[],options);
end
