function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;


%pval: validation set, 
%yval: ground trueth
predictions=zeros(size(pval));

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

predictions ((pval < epsilon))=1; %this vector is only 1 for the outliers


%calculate F1 score

%precision for a class is the number of true positives (i.e. the number of items 
%correctly labeled as belonging to the positive class) 
%divided by the total number of elements labeled as belonging to the positive class 
%(i.e. the sum of true positives and false positives, which are items incorrectly labeled as belonging to the class)


%Recall  is defined as the number of true positives divided
%by the total number of elements that actually belong to the positive class 
%(i.e. the sum of true positives and false negatives, 
%which are items which were not labeled as belonging to the positive class but should have been).

%precison=#true positives/#(true positives + false positives)=# true positives/number of predicted positives

%recall=#true positives/#(true positives +false negatives)=#true positives/ #of actual positives)



%fp= length(find(yval==0 &predictions==1));%number of false positives
fp=sum(yval==0 &predictions==1);
fn=length(find(yval==1 &predictions==0)); %number of false negatives
tp=length(find(yval==1 &predictions==1));%number of true posives

precision=tp/(tp+fp);
recall=tp/(tp+fn);


F1=2*precision*recall/(precision+recall);



    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
