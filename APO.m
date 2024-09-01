function [all_sub_metrics, all_metrics, APO_weights, concatenated_weights] = APO(df_returns,r_bar)

% APO - Adaptive Portfolio Optimization
% This function performs model selection from Lasso, Ridge, and 
% Elastic Net regression models and evaluates the portfolio performance.
%
% Inputs:
%   - df_returns: return data 
%   - r_bar: target return used for the optimization process
%
% Outputs:
%   - all_sub_metrics: Performance metrics for each submission period
%   - all_metrics: Overall performance metrics for the entire period
%   - APO_weights: Optimal weights derived from the optimization models
%   - concatenated_weights: Combined weights across all periods


% For benchmark strategies, define target dates for slicing data
target_dates={
    '2022/03/06'; '2022/04/03'; '2022/05/01'; '2022/05/29'; '2022/06/26';
    '2022/07/24'; '2022/08/21'; '2022/09/18'; '2022/10/16'; '2022/11/13';
    '2022/12/11'; '2023/01/08'
};

% Initialize a cell array to store the extracted data
all_extracted_data=cell(1,numel(target_dates));

in_sample_size=252;

% Loop over each target date
for i=1:numel(target_dates)

    target_date=datetime(target_dates{i},'InputFormat','yyyy/MM/dd');
    
    df_returns.Date=datetime(df_returns.Date,'InputFormat','yyyy/MM/dd');
    
    first_row_after_date=find(df_returns.Date>target_date,1,'first');
    
    if ~isempty(first_row_after_date)

        start_idx=max(1,first_row_after_date-in_sample_size);

        all_extracted_data{i}=df_returns(start_idx:first_row_after_date-1,:);

    else

        all_extracted_data{i}=[];
    end
end

% Initialize cells to store training and validation sets
all_X=cell(size(all_extracted_data));

all_X_train=cell(size(all_extracted_data));

all_X_val=cell(size(all_extracted_data));

% Prepare training and validation sets from the extracted data
for i=1:numel(all_extracted_data)

    all_extracted_data{i}(:,1)=[]; % Remove 'Date' column

    all_X{i}=table2array(all_extracted_data{i});

    all_X_train{i}=all_X{i}(1:189,:);

    all_X_val{i}=all_X{i}(190:end,:);
end

% Compute optimal weights for all submissions
APO_weights_all_submissions=cell(1, numel(all_X_train));

for i=1:numel(all_X_train)

    X_train_val=all_X{i};

    X_train=all_X_train{i};

    X_val=all_X_val{i};

    y_train_val=repmat(r_bar, size(X_train_val, 1), 1);

    y_train=repmat(r_bar, size(X_train, 1), 1);

    y_val=repmat(r_bar, size(X_val, 1), 1);

    num_val_sample = length(y_val);

    % Define models
    models = {'lasso', 'ridge', 'enet'};

    lambda_values = [0.1, 0.01, 0.001, 0.0001];

    % Initialize best model variables
    best_model = '';
    best_lambda = 0;
    max_error = inf;

    % Find the best model and lambda
    for j = 1:length(models)

        model_name=models{j};

        for lambda=lambda_values

            optimal_coefficients=get_optimal_coeff(X_train,y_train,model_name,lambda);

            error=sum((y_val-X_val*optimal_coefficients).^2)/num_val_sample;

            if error<max_error

                best_model=model_name;

                best_lambda=lambda;

                max_error=error;
            end
        end
    end

    % Train final model with best parameters
    optimal_coefficients=get_optimal_coeff(X_train_val,y_train_val,best_model,best_lambda);

    APO_weights_all_submissions{i}=optimal_coefficients';

end

APO_weights=cell2mat(APO_weights_all_submissions');

% Define target dates for global performance evaluation
target_dates = {
    '2022/03/06'; '2022/04/03'; '2022/05/01'; '2022/05/29'; '2022/06/26';
    '2022/07/24'; '2022/08/21'; '2022/09/18'; '2022/10/16'; '2022/11/13';
    '2022/12/11'; '2023/01/08'; '2023/02/18'
};

% Split returns data between defined intervals
df2_returns=df_returns;

df2_returns(:,1)=[];

returns=table2array(df2_returns);

rows_between_intervals=zeros(1,numel(target_dates)-1);

split_returns=cell(1,size(all_X,2));

for i=1:numel(target_dates)-1

    target_start=datetime(target_dates{i}, 'Format', 'yyyy/MM/dd');

    target_end=datetime(target_dates{i+1}, 'Format', 'yyyy/MM/dd');

    indices=df_returns.Date>=target_start & df_returns.Date<target_end;

    rows_between_intervals(i)=sum(indices);

    split_returns{i}=returns(indices,:);

end

% Calculate submission metrics for each interval
all_sub_metrics=zeros(size(all_X,2),3);

for i=1:size(all_X,2)

    submission_data=split_returns{i};

    K=size(submission_data,1);

    result=zeros(K,1);

    for k=1:K

        result(k)=sum(APO_weights(i,:).*submission_data(k,:));
    end

    [sum_ret,stdev,IR]=performance2(result);

    all_sub_metrics(i,:)=[sum_ret,stdev,IR];

end

% Concatenate weights across all intervals
concatenated_weights=[];

for i=1:numel(rows_between_intervals)

    weights=squeeze(APO_weights(i,:));

    repeated_weight=repmat(weights,rows_between_intervals(i),1);

    concatenated_weights=[concatenated_weights;repeated_weight];

end

% Evaluate overall performance metrics
df_returns(:,1)=[];

returns=table2array(df_returns);

p=diag(returns(in_sample_size+1:end,:)*concatenated_weights');

[sum_ret,stdev,IR]=performance2(p);

all_metrics=[sum_ret,stdev,IR]';

end