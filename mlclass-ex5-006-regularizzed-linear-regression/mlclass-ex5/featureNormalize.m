function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);
%X_norm2=X-mu;%gives exactly the same result as X_norm

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);%rdivide is right array divide


% ============================================================

end
