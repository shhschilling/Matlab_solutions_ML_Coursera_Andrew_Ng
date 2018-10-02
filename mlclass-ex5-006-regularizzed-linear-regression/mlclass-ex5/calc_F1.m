function F1 = calc_F1(TP, TN, FP, FN)
%
%   F1 = calc_F1(TP, TN, FP, FN) computes F1
%with
%TP: true positives
%TN: true negatives ..
%FP: False positives
%FN: False negatives


%accuracy = (TP + TN) / (TP+TN+FP+FN)
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
F1 = (2 * precision * recall) / (precision + recall);

%end
