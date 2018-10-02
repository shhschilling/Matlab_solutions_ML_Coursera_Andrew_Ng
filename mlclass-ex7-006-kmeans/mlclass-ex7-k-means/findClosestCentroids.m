function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); %number of centroids

% You need to return the following variables correctly.
m=size(X,1) ;
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
%number of examplesX

idxk=zeros(K,m);
for i=1:m %for loop over examples 

    for j=1:K %for loop over number of clusters
        
    idxk(j,i)=norm(X(i,:)-centroids(j,:))^2;
   

    end


% =============================================================

end
[C,I]=min(idxk);
idx=I';

