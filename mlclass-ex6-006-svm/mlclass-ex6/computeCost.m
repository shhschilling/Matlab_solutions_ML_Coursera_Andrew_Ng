function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0; %initialization of J
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% sumi=zeros(m,1);%initialization of each individual sum term
% for i=1:m
% 
% sumi(i)= (dot(theta',X(i,:))-y(i,:))^2;
% % =========================================================================
% 
% end
% J=sum(sumi(:))/(2*m);



J=(X*theta-y)'*(X*theta-y)/(2*m);