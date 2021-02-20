function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%



h = sigmoid(X * theta);
A=log(h);
A=A.*(-1);
T1=A'*y; %' left terminal

B=ones(size(y));
B=B-y;
C=zeros(size(h));
C = C .+ 1;
C = C + (h .* (-1));
C = log(C);
T2 = B'*C; %' this should give term2
T = T1-T2;
J=T/m;

grad=((X' * (h-y)))/m; %'grad equation


% =============================================================

end
