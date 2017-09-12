function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% size(theta) == 3x1
% size(X) == 100x3
% size(y) == 100x1

% Cost function J, where size(J) == 1x1
% must have size(h) == size(y) == 100x1, b/c we do h-y later
h = sigmoid(X*theta);  % (100x3)*(3x1) == 100x1

diff1 = -y .* log(h);  % (100x1).*(100x1) = 100x1
diff2 = (1-y) .* log(1-h);  % (100x1).*(100x1) = 100x1

new_theta = theta(2:end);  % Don't include theta_0
cost_sigma = (lambda/(2*m)) .* sum(new_theta.^2);

J = (1/m) * sum(diff1 - diff2) + cost_sigma;  % size( sum((100x1)) ) == 1x1


% Partial derivative grad, where size(grad) == 3x1
grad_sigma = (lambda/m) * new_theta;
grad_sigma = [0; grad_sigma];
sum_term = h - y;  % size(sum_term) == 100x1

grad = (1/m) * (X'*sum_term) + grad_sigma;  % (2x1) .* (3x100)*(100x1) = 3x1



% =============================================================

grad = grad(:);

end
