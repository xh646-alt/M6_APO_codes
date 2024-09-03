% Main Execution Script for Portfolio Optimization and Performance Evaluation

% === Data Loading and Preparation for M6 data set===
opts=detectImportOptions('new_df_return.csv');

opts.VariableNamingRule='preserve'; % Preserve variable names

df_returns=readtable('new_df_return.csv',opts);

df_returns.Properties.VariableNames{1}='Date'; % Rename first column to 'Date'

df_returns.Date=datetime(df_returns.Date,'InputFormat','yyyy/MM/dd'); % Convert 'Date' column to datetime

[all_sub_metrics,all_metrics,~,~] = APO(df_returns);



% Load S&P500 returns
data=readtable('filtered_return.csv');    % This csv file contain the s&p500 constituent stocks

load sp25.mat   % This mat file contain the indices of which stock we randomly sample, change to sp50.mat or sp100.mat for additional datasets

df_returns=data(:,sampled_indices);

df_returns.Properties.VariableNames{1}='Date';

df_returns(1,:)=[];

df_returns.Date=datetime(df_returns.Date,'InputFormat','yyyy/MM/dd');

[APO_sub_metrics,APO_metrics,xt_APO,concatenated_APO] = APO(df_returns);

% For benchmark strategies, define target dates for slicing data
target_dates={
    '2022/03/06'; '2022/04/03'; '2022/05/01'; '2022/05/29'; '2022/06/26';
    '2022/07/24'; '2022/08/21'; '2022/09/18'; '2022/10/16'; '2022/11/13';
    '2022/12/11'; '2023/01/08'
};

% Initialize a cell array to store the extracted data
all_extracted_data=cell(1,numel(target_dates));

% Loop over each target date
for i=1:numel(target_dates)

    target_date=datetime(target_dates{i},'InputFormat','yyyy/MM/dd');
    
    df_returns.Date=datetime(df_returns.Date,'InputFormat','yyyy/MM/dd');
    
    first_row_after_date=find(df_returns.Date>target_date,1,'first');
    
    if ~isempty(first_row_after_date)

        start_idx=max(1,first_row_after_date-252);

        all_extracted_data{i}=df_returns(start_idx:first_row_after_date-1,:);

    else

        all_extracted_data{i}=[];
    end
end

% Initialize a cell array to store the modified matrices
all_X=cell(size(all_extracted_data));

% Loop over each table in all_extracted_data
for i=1:numel(all_extracted_data)

    all_extracted_data{i}(:,1)=[];
    
    all_X{i}=table2array(all_extracted_data{i});
end

% === Strategy Implementation ===
num_hold=size(all_X,2);

N=size(xt_APO,2);
% weights for naive diversification
x=(1/N)*ones(1,N);  

xt_naive=repmat(x,num_hold,1);


% weights for Volatility timing strategy:
xt_VT=zeros(num_hold,N);

for i=1:num_hold
    y=all_X{i};
    xt_VT(i,:)=volatility_timing(y,1);
end


% weights for constrained minimum_variance
xt_minvar=zeros(num_hold,N);

for i=1:num_hold
    y=all_X{i};
    xt_minvar(i,:)=minvar(y,1);
end


% weights for constrained mean-variance
xt_meanvar=zeros(num_hold,N);

for i=1:num_hold
    y=all_X{i};
    xt_meanvar(i,:)=meanvar(y,1,1);
end


% weights for Black-Litterman
xt_BL=zeros(num_hold,N);

for i=1:num_hold
    y=all_X{i};
    xt_BL(i,:)=Black_Litterman(y,1,1);
end


% weights for Bayes-Stein
xt_BS=zeros(num_hold,N);
K=252;

for i=1:num_hold
y=all_X{i};
xt_BS(i,:)=Bayes_Stein(y,K,1,1);
end


% Define dates within holding period for global performance evaluation
target_dates={
    '2022/03/06'; '2022/04/03'; '2022/05/01'; '2022/05/29'; '2022/06/26';
    '2022/07/24'; '2022/08/21'; '2022/09/18'; '2022/10/16'; '2022/11/13';
    '2022/12/11'; '2023/01/08'; '2023/02/18'
};

rows_between_intervals=zeros(1,numel(target_dates)-1);

% Loop over each interval in target_dates
for i=1:numel(target_dates)-1

    target_start=datetime(target_dates{i},'Format','yyyy/MM/dd');

    target_end=datetime(target_dates{i+1},'Format','yyyy/MM/dd');
    
    interval_rows=df_returns.Date>=target_start&df_returns.Date<target_end;
    
    rows_between_intervals(i)=sum(interval_rows);
end


xt_all=cat(3,xt_naive,xt_VT,xt_minvar,xt_meanvar,xt_BS,xt_BL,xt_APO);

concatenated_weights=zeros(241,N,size(xt_all,3));

% Loop over each choice of weights
for j = 1:size(xt_all,3)

    concatenated_weights_temp=[];

    % Loop over each weight vector and corresponding rows_between_intervals value
    for i = 1:numel(rows_between_intervals)

        weights_j=squeeze(xt_all(i,:,j));

        repeated_weight=repmat(weights_j,rows_between_intervals(i),1);

        concatenated_weights_temp=[concatenated_weights_temp; repeated_weight];
    end

    concatenated_weights(:,:,j)=concatenated_weights_temp;
end

df_returns(:,1)=[];

returns=table2array(df_returns);

in_sample_size=252;

all=zeros(3,size(xt_all,3));

p_all=zeros(241,size(xt_all,3));

for i=1:size(xt_all,3)

	p_all(:,i)=diag(returns(in_sample_size+1:end,:)*concatenated_weights(:,:,i)');

	p=p_all(:,i);

	[sum_ret,stdev,IR]=performance2(p);

	all(:,i)=[sum_ret,stdev,IR]';
end


% Performance metrics net of transaction costs
ptc=2;   % 2 basis point, can change to 5 or 10

[post_TC_metrics]=TC(returns,concatenated_weights,ptc);
