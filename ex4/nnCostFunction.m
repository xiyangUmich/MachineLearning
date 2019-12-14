function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda, calcGrad)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m,1) X];
a2 = sigmoid(X*Theta1');
a2 = [ones(m,1) a2];
h = sigmoid(a2*Theta2');
ym = zeros(m,num_labels);
for i=[1:m]
    ym(i,y(i,1)) = 1;
end

J_m = (-log(h).*ym - log(1-h).*(1-ym))/m;
J = sum(sum(J_m));
theta1_sum = sum(sum(Theta1(:,2:input_layer_size+1).^2*lambda/(2*m)));
theta2_sum = sum(sum(Theta2(:,2:hidden_layer_size+1).^2*lambda/(2*m)));
J = J + theta1_sum + theta2_sum;

delta3 = h - ym;
delta2 = delta3*Theta2.*a2.*(1-a2);
delta2 = delta2(:, 2:hidden_layer_size+1);
Theta2_grad = delta3'*a2./m;
%T2 = Theta2_grad
Theta1_grad = delta2'*X./m;
Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1)+ Theta1(:, 2:input_layer_size+1).*lambda./m; 
Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1)+ Theta2(:, 2:hidden_layer_size+1).*lambda./m; 
%for i = 1:num_labels,
%   for j = 1:hidden_layer_size+1,
%        for k = 1:m,
%            Theta2_grad(i, j) = Theta2_grad(i, j) + delta3(k, i).*a2(k, j);
%        end;
%	Theta2_grad(i, j) = Theta2_grad(i, j)/m;
%        if j > 1,
%	    Theta2_grad(i, j) = Theta2_grad(i, j) + Theta2(i, j)*lambda;
%	 end;
%    end;
%end;
%for i = 1:hidden_layer_size,
%    for j = 1:input_layer_size+1,
%        for k = 1:m,
%            Theta1_grad(i, j) = Theta1_grad(i, j) + delta2(k, i).*X(k, j);
%        end;
%        Theta1_grad(i, j) = Theta1_grad(i, j)./m;
%        if j > 1,
%            Theta1_grad(i, j) = Theta1_grad(i, j) + Theta1(i, j)*lambda;
%        end;
%    end;
%end;
%T1 = Theta1_grad(1:10, 1:10)
grad = [Theta1_grad(:); Theta2_grad(:)];
end;	



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


%end
