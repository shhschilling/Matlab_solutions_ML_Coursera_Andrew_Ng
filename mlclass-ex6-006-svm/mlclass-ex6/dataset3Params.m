function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%SVMPREDICT returns a vector of predictions using a trained SVM model
%(svmTrain). 
%   pred = SVMPREDICT(model, X) returns a vector of predictions using a 
%   trained SVM model (svmTrain). X is a mxn matrix where there each 
%   example is a row. model is a svm model returned from svmTrain.
%   predictions pred is a m x 1 column of predictions of {0, 1} values.
%

% Check if we are getting a column vector, if so, then assume that we only
% need to do prediction for a single example

Ci=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmai=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
k=0

errork=zeros(1,length(Ci)*length(sigmai));
Ck=zeros(1,length(Ci)*length(sigmai));
sigmak=zeros(1,length(Ci)*length(sigmai));
for i=1:8
Ci(i)
for j=1:8
    k=k+1;
   
model= svmTrain(X, y, Ci(i), @(x1, x2) gaussianKernel(x1, x2, sigmai(j)));
predictions = svmPredict(model, Xval);
errork(k)=mean(double(predictions ~= yval));
Ck(k)=Ci(i);
sigmak(k)=sigmai(j);
end
end
[minu,I]=min(errork);
C=Ck(I);
sigma=sigmak(I);
