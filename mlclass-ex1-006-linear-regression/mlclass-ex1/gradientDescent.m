function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
[m,n] = size(X); % 
J_history = zeros(num_iters, 1);
%calculate vector theta=(theta0,theta1)'

sumi=zeros(m,n); %m samples, n-1 features, added column x0=ones(m,1)
for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % Save the cost J in every iteration
    
    
    
    
    %initialization of each individual sum term
   for j=1:n
        for i=1:m
            
            %sumi(i,1)= (dot(theta',X(i,:))-y(i,:));
            sumi(i,j)= dot((dot(theta',X(i,:))-y(i,:)),X(i,j));
            % =========================================================================
            
        
        
        end
   end
    %guarantees simultanous update
    
             %you have to update both values simultonously within each
             %iteration step
   for j=1:n
    theta(j)=theta(j)-alpha/m *sum(sumi(:,j));
    end
    J_history(iter) = computeCost(X, y, theta);
    
    
    
    
    
    % ============================================================
    
    
    
end

end
