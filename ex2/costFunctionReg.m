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
z = X*theta;
h = sigmoid(z);
l1 = (-1).*log(1-h);
l2 = (-1).*log(h);
J = sum(l1.*(1-y)+l2.*y)/m + (sum(theta.^2)-theta(1)^2)*lambda./(2*m);
grad1 = X'*(h - y)./m;
grad2 = theta*lambda/m;
grad = grad1 + grad2;
grad(1, 1) = grad(1, 1) - grad2(1,1);





% =============================================================

end
