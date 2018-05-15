function C=calConfusionMatrix(predictions,truth)
%       Prediction[     1    0
%Truth              1   TP   FN 
%                   0   FP   TN
%]
%accuracy = TP+TN/(TP+TN+FN+FP)
%Precision = TP/(TP+FP)
%Recall = TP/(TP+FN)
TP = sum(predictions==1 & truth==1);
FP = sum(predictions==1 & truth==0);
TN = sum(predictions==0 & truth==0);
FN = sum(predictions==0 & truth==1);

Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
accuracy = (TP+TN)/(TP+TN+FN+FP)
C = [ TP,FN;...
    FP,TN];

end

