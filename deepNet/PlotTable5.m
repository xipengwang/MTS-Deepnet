clc;
TP=zeros(1,20);
FP=zeros(1,20);
TN=zeros(1,20);
FN=zeros(1,20);
%%
for i=1:6
   tmp = predictions((i-1)*20+1:i*20);
   TP = TP + tmp';
   FN = FN + ~tmp';
end

for i=1:6
    tmp = predictions(120+(i-1)*20+1:120+i*20);
    TN = TN + ~tmp';
    FP = FP + tmp';
end