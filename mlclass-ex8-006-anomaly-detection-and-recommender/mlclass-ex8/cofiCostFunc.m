 function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies,num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%
%Note that the function returns the gradient for both sets of variables by unrolling them into a single vector.
%params: are the features [X;Theta] concatenated_> needed for fmincg


% Unfold the X and Theta matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


%J=(X*Theta-Y)'*(X*Theta-Y)/2;


%Theta - num_users  x num_features matrix of user features=ix k
%Y=num_movies* number of users=i x j=5x4 in example
%X - num_movies  x num_features=i x k
%Theta': num features *num_users=k xj
%k: number of features
%Theta=jxk
%X=ix n



J1=((X*Theta'-Y).^2).*R;

J=sum(sum(J1))/2;

%introduce regularization 
%here we funnily also regularize Theta0!

J=J+lambda/2*sum(sum(Theta.^2))+lambda/2*sum(sum(X.^2))


%Calculation of Theta_grad
%Theta=num_users*numbers of featers

%sum over j=sum overcolumns
X_grad=((X*Theta'-Y).*R) *Theta;
X_grad=X_grad+lambda*X;

%Calculation of Theta_grad
%sum over rows
Theta_grad=((X*Theta'-Y).*R)'*X;
Theta_grad=Theta_grad+lambda*Theta;










% =============================================================

grad = [X_grad(:); Theta_grad(:)];

%end
