function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % size(X) = 97x2
    % size(y) = 97x1
    % size(theta) = 2x1

    % h(x) = theta^T * X, needs to be (97x1) b/c we later do: h - y
    h = X*theta;  % (97x2)*(2x1) = (97x1)
    
    % partial_diff needs to be (2x1), b/c we later substract this from theta
    partial_diff = h - y;  % is (97x1)
    partial_diff = X'*partial_diff;  % (2x97)*(97*1) = (2x1)
    
    theta = theta - (alpha/m)*partial_diff;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
