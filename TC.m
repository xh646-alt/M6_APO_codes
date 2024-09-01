function [all_metrics] = TC(data, xt_all, ptc)

% Function to calculate realized returns after accounting for the porportional transaction costs
%
% Inputs:
%   data   - raw data matrix of asset returns
%   xt_all - concatenated_weights for all models
%   ptc    - porportional transaction costs
%
% Outputs:
%   all_metrics - performance metrics


Estimation=252;
T=size(data,1)-Estimation;

N=size(data,2);
trr=ptc/10000 *ones(1,N);

xxtt=xt_all;

all_metrics=zeros(3,size(xxtt,3));

for D=1:size(xxtt,3)
    xt= xxtt(:,:,D)';
    Portfolio_Returns=zeros(T,1);

    for i=1:T
        Portfolio_Returns(i,1)=data(Estimation+i,1:N)*xt(:,i);
    end

    for i=1:T
        for j=1:N  
        xtt(j,i)=(xt(j,i)*(1+data(Estimation+i,j)))/(1+xt(:,i)'*data(Estimation+i,:)');
        end
    end
    
    S(1)=sum(trr*abs(xt(:,1)));
    for i=1:T-1  
        S(i+1)=sum(trr*abs(xt(:,i+1)-xtt(:,i)));  
    end


    for i=1:T
        Portfolio_Returns(i,1)=Portfolio_Returns(i,1)-S(i);
    end

    p=Portfolio_Returns;

	[sum_ret,stdev,IR]=performance2(p);
	all_metrics(:,D)=[sum_ret,stdev,IR]';

end
