% function [J grad] = nnCostFunction(nn_params, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    num_labels, ...
%                                    X, y, lambda)
%                                
%                                
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

%B = reshape(A,m,n) or B = reshape(A,[m n]) returns the m-by-n matrix B whose elements
%are taken column-wise from A. An error results if A does not have m*n elements.

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));%dimension: number of nodes in hiddenlayer  x number of examples in X+1= dimension of a^(1)
Theta2_grad = zeros(size(Theta2));%number of output nodes   times number of hiddenlayers +1 (=a^2)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%Calculation of cost function J


%rewrite the labels y from mx1 to matrix y=Kxm (K isnum_labels)

Y=zeros(num_labels,m);
 
for j=1:m
Y(y(j),j)=1;

end

%Calculate h(theta)


a1=[ones(m,1) X]; %input layer

z2=Theta1*a1';
a2=sigmoid(z2); %activation function of hidden layer always betweeen 0 1
a2p=[ones(m,1) a2']; %add columns of 1;
z3=Theta2*a2p';
h=sigmoid(z3);
a3=h;
%cost function of neural network with one hidden layer

J=0;


for i=1:m %loop over examples
J=J-Y(:,i)'*log(h(:,i))-(1-Y(:,i)')*log(1-h(:,i));
end
J=J/m;



% you do not have to regularize the first theta column!
%the first colum corresponds to the weights of the added note x0

theta1=Theta1(:,2:end);
theta2=Theta2(:,2:end);
%reg2=sum(sum(Theta1.^2,1),2)+sum(sum(Theta2.^2,1),2)
reg=sum(sum(theta1.^2))+sum(sum(theta2.^2));

reg=reg*lambda/(2*m);

J=J+reg;




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




Deltal1=0;
Deltal2=0;

%1. initialize input layer -> already done abnove    

    
%2. calculate error function of  hidden layer 3

delta3=a3-Y;

%3. calculate error hidden layer 2
%remove Delta02, which corresponds to using theta2
delta2=theta2'*delta3.*sigmoidGradient(z2);


%4.accumulate the gradient of all layers

%should be generalized by for loop over layers; dimensions are not correct!

Deltal2=Deltal2+delta3*a2';
Deltal1=Deltal1+delta2*a1(:,2:end);

Theta1_grad=1/m *Deltal1;
Theta2_grad=1/m *Deltal2;






%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



