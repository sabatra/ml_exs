function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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





h = sigmoid(X * theta);
A=log(h);
A=A.*(-1);
T1=A'*y; %' left terminal

B=ones(size(y));
B=B-y;
C=ones(size(h));
C = C + (h .* (-1));
C = log(C);
T2 = B'*C; %' this should give term2
T = T1-T2;
J=T/m;
theta1=theta;
theta1(1,1)=0;
regl = (lambda/(2*m))*sum(theta1 .^ 2);
J=J+regl;

grad= ( X' * (h-y)  + lambda*theta1 )/m ; %'grad equation


% =============================================================

end
