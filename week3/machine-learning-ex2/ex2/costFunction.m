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

% size(theta) == 3x1
% size(X) == 100x3
% size(y) == 100x1

% Cost function J, where size(J) == 1x1
% must have size(h) == size(y) == 100x1, b/c we do h-y later
h = sigmoid(X*theta);  % (100x3)*(3x1) == 100x1

diff1 = -y .* log(h);  % (100x1).*(100x1) = 100x1
diff2 = (1-y) .* log(1-h);  % (100x1).*(100x1) = 100x1

J = (1/m) * sum(diff1 - diff2);  % size( sum((100x1)) ) == 1x1

% Partial derivative grad, where size(grad) == 3x1
sum_term = h - y;  % size(sum_term) == 100x1
grad = (1/m) * (X'*sum_term);  % (1x1) .* (3x100)*(100x1) = 3x1

% =============================================================

end
