%%
% Program Main Entrance

%% read data
clear;close all;
path = 'C:\Users\Xipeng1990\Desktop\Thesis\program\dataOutput\ID13_dev\dataSynchronizationOutput\';
data = readData(path);   

%% data normalization
type = 1;
% normalization type;
% if type = 1, which [0,1] normalization; final_active_func_option should be 2.
% if type = 2, which is [-1,1] normalization; final_active_func_option should be 3.
for dummy=1:1
    numOfPositive = size(data.positive,2);
    numOfNegative = size(data.negative,2);
    randIdx = [1:numOfNegative + numOfPositive];
    featureMatrix = [];
    for i=1:(numOfNegative + numOfPositive)
        if(randIdx(i)<=numOfPositive)
            featureMatrix = [featureMatrix; data.positive{1,randIdx(i)}];
        else
            featureMatrix = [featureMatrix; data.negative{1,randIdx(i)-numOfPositive}];
        end
    end
    % maxVec = max(featureMatrix);
    % minVec = min(featureMatrix);
    
    %1          2           3           4           5           6           7
    %Target     EcgRaw      HR          RR          GsrRaw      GSR         SCL
    %8          9           10          11          12          13          14
    %SCR        RspRaw      Exp Vol     Insp Vol    qDEEL       Resp Rate   Resp Rate Inst
    %15         16          17          18          19          20          21
    %Te         Ti          Ti/Te       Ti/Tt       Tpef/Te     Tt          Vent
    %22         23
    %Vt/Ti      Work of Breathing
    
    %         1     2     3   4     5    6    7     8    9
    maxVec = [1,  700,  120,  1,  500,  10,  10,  0.3,  700, ...
        2800, 2200,  800, 100,  40, 4,  10, 10, 3,  1,  10, 60, 2200, 1300];
    %   10      11     12   13  14  15  16  17  18  19   20  21  22    23
    
    %         1     2     3   4     5    6    7     8    9
    minVec = [0,  350,  60,  0.5,  120,  2,   2,  -0.5,  250, ...
        -500, -400,  -100, -30,  2, -2,  -1, -3, 0,  -1,  -1, -20, -700, -600];
    %   10      11     12   13   14 15   16  17  18  19   20   21   22    23


maxMatrix = ones(size(data.positive{1,1},1),1)*maxVec;
minMatrix = ones(size(data.positive{1,1},1),1)*minVec;
if(type==1)
    for i=1:(numOfNegative + numOfPositive)
        if(randIdx(i)<=numOfPositive)
            data.positive{1,randIdx(i)} = (data.positive{1,randIdx(i)} - minMatrix)./(maxMatrix - minMatrix);
        else
            data.negative{1,randIdx(i)-numOfPositive} = (data.negative{1,randIdx(i)-numOfPositive} - minMatrix)./(maxMatrix - minMatrix);
        end
    end
else
    for i=1:(numOfNegative + numOfPositive)
        if(randIdx(i)<=numOfPositive)
            data.positive{1,randIdx(i)} = -1 + 2*(data.positive{1,randIdx(i)} - minMatrix)./(maxMatrix - minMatrix);
        else
            data.negative{1,randIdx(i)-numOfPositive} = -1 + 2*(data.negative{1,randIdx(i)-numOfPositive} - minMatrix)./(maxMatrix - minMatrix);
        end
    end
end
end
save('data.mat');
%% training
[posIdx]=randperm(length(data.positive));
[negIdx]=randperm(length(data.negative));

%first 10 positive as testing;first 20 negative as testing
teData.positive = data.positive(posIdx(1:10));
teData.negative = data.negative(negIdx(1:10));
trData.positive = data.positive(posIdx(11:62));
trData.negative = data.negative(negIdx(11:62));

% trPara.netStruct = [20  5   10  5   5   5   3   5   1  10  1;...
%     22  5   5   4   4   4   4   3   3   0  0];
% trPara.netStruct = [20 15  10  3   5   2   1   5  1;...
%     22  2   2   2   2   1   1   0   0];

trPara.netStruct = [20   15   1   5   1;...
                    22   2    2   0   0];

% trPara.netStruct = [20 10 5  5  1  5  1;...
%                    22  2  2  2  2  0  0];


                
trPara.lr = 0.1;
trPara.epoch = 200;
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
save([datestr(now,'yyyy_mm_dd_HH_MM_SS') '.mat']);

