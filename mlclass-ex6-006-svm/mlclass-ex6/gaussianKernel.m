function sim = gaussianKernel(x1, x2, sigma)

%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim
%x1: training example
%x2: landmarls
% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
%similarity is 1 if x1==x2
sim=exp(-(norm(x1-x2))^2/(2* sigma^2));




% =============================================================
    
end
