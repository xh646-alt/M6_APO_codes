function w = get_optimal_coeff(X_train, y_train, model, lambda)

% This function computes the optimal coefficients for a given regression model (lasso, ridge, or elastic net).
%
% Inputs:
%   X_train - Training data matrix (features)
%   y_train - Training response vector
%   model   - Model type ('lasso', 'ridge', 'enet')
%   lambda  - Regularization parameter
%
% Output:
%   w - Optimal coefficients for the specified model


% Validate inputs
validateattributes(X_train,{'numeric'},{'2d'},mfilename,'X_train',1);

validateattributes(y_train,{'numeric'},{'column'},mfilename,'y_train',2);

validateattributes(lambda,{'numeric'},{'scalar','nonnegative'},mfilename,'lambda',4);

% Ensure the model type is valid
validModels = {'lasso','ridge','enet'};

if ~ismember(model,validModels)

    error('Invalid model type. Choose ''lasso'', ''ridge'', or ''enet''.');
end

% Select the appropriate model based on the input
switch model
    case 'lasso'
        % Compute optimal coefficients using Lasso regression
        w=lasso_mma(X_train,y_train,lambda);

    case 'ridge'
        % Compute optimal coefficients using Ridge regression
        w=ridge_mma(X_train,y_train,lambda);

    case 'enet'
        % Compute optimal coefficients using Elastic Net regression
        w=enet_mma(X_train,y_train,lambda);
        
    otherwise
        % This should never be reached due to the input validation
        error('Unexpected model type.');
end

end