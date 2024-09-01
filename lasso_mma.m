function [w]=lasso_mma(X_train,y_train,lambda)

% Function to perform Lasso regression using the Moving Asymptotes (MMA) optimization method
%
% Inputs:
%   X_train - training data matrix, where each row is an observation and each column is a feature
%   y_train - training response vector
%   lambda  - regularization parameter that controls the sparsity of the solution
%
% Outputs:
%   w  - vector of optimized weights


% Initialize variables
NumW=size(X_train,2);
w=ones(NumW,1);

m=3;
NumT=size(X_train,1);

ConsFac=ones(NumW,1);
ObjFac=ones(NumT,1);

zerom   = zeros(m,1);
wold1   = w;
wold2   = wold1;
eeen    = ones(NumW,1);
eeem    = ones(m,1);
wmin    = w*0;
wmax    = w*0+1;
low     = wmin;
upp     = wmax;
c       = 1000*eeem;
d       = eeem;
a0      = 1;
a       = zerom;

% Iteration parameters
Iter=1;
maxIter=100;
tau=0.99999;         % Constraint threshold
move=0.02;           % Step size for weight adjustments

% Initialize vectors to store previous weights and objective values
wold1=w;
wold2=wold1;
objective_summarize=zeros(maxIter,3); % Stores objective values, changes, and constraints

% Start the gradient descent loop
while Iter<maxIter

    NumT_temp=1;

    residual=y_train-X_train*w;

    % Objective function and its gradient
    ObjFun=ObjFac'*(residual.^2)/NumT_temp+lambda*(ConsFac'*abs(w));

    ObjFun_diff=(-2/NumT_temp*residual'*X_train)'+lambda*ConsFac.*sign(w);

    % Constraint function and its gradient
    ConsFun=(ConsFac'*w-tau);

    ConsFun_diff=ConsFac;

    % Save raw objective and constraint values for scaling
    f_raw=ObjFun;
    dfdw_raw=ObjFun_diff;

    g_raw=ConsFun;
    dgdw_raw=ConsFun_diff';

    if mod(Iter-1,20)==0
        f_Ini=abs(f_raw);
        g_Ini=abs(g_raw);
    end

    % Convergence criteria based on constraint and change in weights
    if abs(g_raw(1))<1e-5 && Iter>1 && max(abs(wold2-wold1))<1e-5
        break;
    end

    % Normalization of objective and constraint for MMA optimization
    [f,dfdw]=deal(f_raw/f_Ini,dfdw_raw/f_Ini);
    [g,dgdw]=deal(g_raw./g_Ini,dgdw_raw./g_Ini);

    % Define the move limits for weights
    wmin_mma=max(w-move,wmin);
    wmax_mma=min(w+move,wmax);

    m=1;
    Cons_Index=1;

    [wmma,~,~,~,~,~,~,~,~,~,~]=mmasub(m,NumW,Iter,w,wmin_mma,wmax_mma,...
        wold1,wold2,f,dfdw,g(Cons_Index),dgdw(Cons_Index,:),low,upp,a0,a(Cons_Index,1),c(Cons_Index,1),d(Cons_Index,1));

    % Update weights and store the previous values
    w=wmma;
    wold2=wold1;
    wold1=w;

    Chg=max(abs(wold2-wold1));

    % Display iteration information
    disp(['Iter: ',num2str(Iter),'; Chg: ',num2str(Chg),...
          '; ConsFun: ',num2str(ConsFun),'; Sum:',num2str(sum(w))]);

    % Store objective function value, change, and constraint function value
    objective_summarize(Iter,:)=[ObjFun,Chg,ConsFun];

    % Increment the iteration counter
    Iter=Iter+1;
end

end