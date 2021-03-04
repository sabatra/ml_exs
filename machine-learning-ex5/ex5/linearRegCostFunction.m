function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% =========================================================================

h = X * theta;
errorRate = h-y;
sqdError = errorRate .^2;
J = (1/(2*m)) * sum(sqdError);
sqtheta = theta(2:end) .^2;
regu = ((lambda/(2*m)) * sum(sqtheta));
J = J + regu;

grad = errorRate'*X; %'using metrics
grad = grad .* (1/m);
thetareg = theta;
thetareg(1)=0;
thetareg = thetareg .* (lambda/m);
grad = grad + thetareg'; %'issue is transpose
grad = grad(:);

end
