 %%
% Program Main Entrance

%% read data
clear;close all;
path = 'C:\Users\XIPENG\Desktop\Thesis\program\MITdata\old\';
files = dir(path);
dataIdx = [7:20];
posk=1;
negk=1;
for i=3:length(files)-3
    load([path files(i).name]);
    for j=20:size(dataAll,1)
        if(dataAll(j,67)>=50)
           target = zeros(20,1);
           trData.negative{negk} = [target,dataAll(j-19:j,dataIdx)];
           negk = negk + 1;
        else
           target = ones(20,1);
           trData.positive{posk} = [target,dataAll(j-19:j,dataIdx)]; 
           posk = posk + 1;
        end
    end
end
posk=1;
negk=1;
for i=length(files)-2:length(files)
    load([path files(i).name]);
    for j=20:size(dataAll,1)
        if(dataAll(j,67)>=30)
           target = zeros(20,1);
           teData.negative{negk} = [target,dataAll(j-19:j,dataIdx)];
           negk = negk + 1;
        else
           target = ones(20,1);
           teData.positive{posk} = [target,dataAll(j-19:j,dataIdx)]; 
           posk = posk + 1;
        end
    end
end
%% data normalization
type = 1;
% normalization type;
% if type = 1, which [0,1] normalization; final_active_func_option should be 2.
% if type = 2, which is [-1,1] normalization; final_active_func_option should be 3.
%% training
%first 10 positive as testing;first 20 negative as testing

% trPara.netStruct = [20  5   10  5   5   5   3   5   1  10  1;...
%     22  5   5   4   4   4   4   3   3   0  0];
% trPara.netStruct = [20 15  10  3   5   2   1   5  1;...
%     22  2   2   2   2   1   1   0   0];

trPara.netStruct = [20   15   1   5   1;...
                    14   1    1   0   0];


  
trPara.lr = 0.1;
trPara.epoch = 5000;
trPara.performance = 1E-8;
trPara.gradientCheck = 1E-5;
trPara.active_func_option = 2;
if(type==1)
    trPara.final_active_func_option = 2;
else
    trPara.final_active_func_option = 3;
end
[net] = deepTrain(trData,trPara);
%%
[tr_predictions,tr_truth] = deepTest(trData,net);
[tr_confusionMatrix] = calConfusionMatrix(tr_predictions,tr_truth)
%% testing
[predictions,truth] = deepTest(teData,net);
[confusionMatrix] = calConfusionMatrix(predictions,truth)
%%
save([datestr(now,'yyyy_mm_dd_HH_MM_SS') '_alertProj.mat']);

