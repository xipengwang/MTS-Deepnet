%%
% Program Main Entrance without read data, but load data from data.mat

%% read data
clear;close all;
% path = 'C:\Users\XIPENG\Desktop\Thesis\program\dataOutput\ID13_dev\dataSynchronizationOutput\';
% data = readData(path);
load('data.mat');
%% data normalization
type = 1;
% normalization type;
% if type = 1, which [0,1] normalization; final_active_func_option should be 2.
% if type = 2, which is [-1,1] normalization; final_active_func_option should be 3.
%% training N folder validation
addpath('../minFunc');
Nfolder  = 10;
lenData = length(data.positive)/20;
[posIdx]=randperm(lenData);
[negIdx]=randperm(lenData);
posIdxCell = cell(2,Nfolder);
negIdxCell = cell(2,Nfolder);
for i=1:Nfolder
   tmp =  posIdx;
   stepLen = floor(lenData/Nfolder);
   tmpIdx = tmp((i-1)*stepLen+1:i*stepLen);
   for j=1:length(tmpIdx)
        posIdxCell{1,i} = [posIdxCell{1,i},(tmpIdx(j)-1)*20+1:tmpIdx(j)*20];   
   end
   tmp((i-1)*stepLen+1:i*stepLen) = [];
   tmpIdx = tmp;
   for j=1:length(tmpIdx)
        posIdxCell{2,i} = [posIdxCell{2,i},(tmpIdx(j)-1)*20+1:tmpIdx(j)*20];   
   end
   
   tmp =  negIdx;
   stepLen = floor(lenData/Nfolder);
   tmpIdx = tmp((i-1)*stepLen+1:i*stepLen);
   for j=1:length(tmpIdx)
        negIdxCell{1,i} = [negIdxCell{1,i},(tmpIdx(j)-1)*20+1:tmpIdx(j)*20];   
   end
   tmp((i-1)*stepLen+1:i*stepLen) = [];
   tmpIdx = tmp;
   for j=1:length(tmpIdx)
        negIdxCell{2,i} = [negIdxCell{2,i},(tmpIdx(j)-1)*20+1:tmpIdx(j)*20];   
   end
   
end
clear tmp;clear tmpIdx;

for i=10:Nfolder
    %first 10 positive as testing;first 20 negative as testing
    teData.positive = data.positive(posIdxCell{1,i});
    teData.negative = data.negative(negIdxCell{1,i});
    trData.positive = data.positive(posIdxCell{2,i});
    trData.negative = data.negative(negIdxCell{2,i});
    
    % trPara.netStruct = [20  5   10  5   5   5   3   5   1  10  1;...
    %     22  5   5   4   4   4   4   3   3   0  0];
    % trPara.netStruct = [20 15  10  3   5   2   1   5  1;...
    %     22  2   2   2   2   1   1   0   0];
    
    trPara.netStruct = [20 15 1  5  1;...
                       22  3  3  0  0];
    
%     trPara.netStruct = [20 10 5  5  1  5  1;...
%                        22  1  1  1  1  0  0];
    
    
    trPara.lr = 0.1;
    trPara.epoch = 2000;
    trPara.performance = 1E-8;
    trPara.gradientCheck = 1E-5;
    trPara.active_func_option = 2;
    if(type==1)
        trPara.final_active_func_option = 2;
    else
        trPara.final_active_func_option = 3;
    end
    %%
    [net] = deepTrain(trData,trPara);
    %% training
    [tr_predictions,tr_truth] = deepTest(trData,net);
    [tr_confusionMatrix] = calConfusionMatrix(tr_predictions,tr_truth)
    %% testing
    [predictions,truth] = deepTest(teData,net);
    [confusionMatrix] = calConfusionMatrix(predictions,truth)
    %%
    save([datestr(now,'yyyy_mm_dd_HH_MM_SS') '_folder_' num2str(i) '.mat']);
end

