%% Deep network training
function [net] = deepTrain(trData,trPara)
%% Parameters
%trData: data.positive(lane change events) and data.negative(lane keeping events)

%trPara.

%net

%% initial network
for dummy=1:1
    netStruct = trPara.netStruct;
    %netStruct = [N  K2   K3    ... Kn   h  1...
    %             M  L2   L3    ... Ln   0  0];
    % even layer:   convolutional layer
    % odd layer:    down-sample layer
    layerNum = size(netStruct,2);
    net.layer = cell(size(layerNum,2),1);
    % check parameters
    N = netStruct(1,1);
    M = netStruct(2,1);
    % assert( size(trData.positive{1},1)==N && size(trData.positive{1},2)==M ...
    %     && size(trData.negative{1},1)==N && size(trData.negative{1},2)==M, ...
    %     'Error netStruct, check first column');
    
    tmp = N;
    for i = 2:2:layerNum-3
        tmp = (tmp - netStruct(1,i)+1);
        assert(tmp>0, 'Check convolutional layer parameters!')
    end
    clear tmp;
    
    tmpN = N;
    tmpM = M;
    dimMatrix = zeros(size(netStruct));
    for i=1:size(netStruct,2)-2
        dimMatrix(1,i) = tmpN;
        dimMatrix(2,i) = tmpM;
        if(mod(i,2)==1)
            tmpN = tmpN - netStruct(1,i+1) + 1;
        else
            tmpM = netStruct(1,i+1);
        end
    end
    clear tmpN tmpM;
    dimMatrix(1,i+1)=netStruct(1,i+1);
    dimMatrix(2,i+1)=1;
    dimMatrix(1,i+2)=netStruct(1,i+2);
    dimMatrix(2,i+2)=1;
    
    elementNum = 1;
    pre_elementNum = 0;
    for i=1:size(netStruct,2)
        if(elementNum==0) %last layer only has one cell
            net.layer{i} = cell(1,1);
        else
            net.layer{i} = cell(elementNum,1);
        end
        if(i==1) %first layer
            s.a = zeros(N,M);
            net.layer{i}{1} = s;
        elseif(i==size(netStruct,2)) %final layer
            s.a = 0;
            s.z = 0;
            s.w  = randomCovWeights(1,dimMatrix(1,i-1));
            s.w_delta=zeros(1,dimMatrix(1,i-1));
            s.b  = randomCovWeights(1,1);
            s.b_delta = zeros(1,1);
            s.errorterm = 0;
            net.layer{i}{1} = s;
        elseif(i==size(netStruct,2)-1) %final hidden layer
            s.a = zeros(dimMatrix(1,i),dimMatrix(2,i));
            s.z = zeros(dimMatrix(1,i),dimMatrix(2,i));
            s.errorterm = zeros(dimMatrix(1,i),dimMatrix(2,i));
            s.w  = randomCovWeights(dimMatrix(1,i),dimMatrix(1,i-1)*pre_elementNum);
            s.w_delta=zeros(dimMatrix(1,i),dimMatrix(1,i-1)*pre_elementNum);
            s.b  = randomCovWeights(dimMatrix(1,i),1);
            s.b_delta = zeros(dimMatrix(1,i),1);
            net.layer{i}{1} = s;
        else %other layers
            for j = 1:size(net.layer{i},1)
                s.a = zeros(dimMatrix(1,i),dimMatrix(2,i));
                s.z = zeros(dimMatrix(1,i),dimMatrix(2,i));
                s.errorterm = zeros(dimMatrix(1,i),dimMatrix(2,i));
                if(mod(i,2)==0) %convolutional layer
                    s.w = randomCovWeights(netStruct(1,i),dimMatrix(2,i));
                    s.w_delta = zeros(netStruct(1,i),dimMatrix(2,i));
                    s.b = randomCovWeights(1,dimMatrix(2,i));
                    s.b_delta = zeros(1,dimMatrix(2,i));
                else %down-sample layer
                    s.w = randomDownWeights(dimMatrix(2,i),dimMatrix(2,i-1),dimMatrix(1,i));
                    s.w_delta = zeros(dimMatrix(2,i),dimMatrix(2,i-1),dimMatrix(1,i));
                    s.b = randomDownWeights(dimMatrix(2,i),1,dimMatrix(1,i));
                    s.b_delta = zeros(dimMatrix(2,i),1,dimMatrix(1,i));
                end
                net.layer{i}{j} = s;
            end
        end
        pre_elementNum = elementNum;
        if(mod(i,2)==1 && elementNum~=0)
            elementNum = elementNum * netStruct(2,i+1);
        end
    end
    weightsVector = net2vector(net);
end
%% training
net.cost = [];
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
options.maxIter = trPara.epoch;	  % Maximum number of iterations of L-BFGS to run
options.MaxFunEvals = 2000;
options.display = 'on';
% options.LS_init = 2;

% options.alpha = 0.1;
% options.minibatch = 100;
% options.epochs = trPara.epoch;
tic;
[opttheta, net.cost] = minFunc(@(p) deepCost(p,net,trData,trPara), ...
        weightsVector, options);
% [opttheta,net.cost] = minFuncSGD(@(p,data) deepCost(p,net,data,trPara), ...
%         weightsVector, trData, options);
weightsVector = opttheta;
trainingTime=toc;
% for epoch = 1:trPara.epoch
%     fprintf('Current epoch number: %d \n',epoch);
%     tic;
%     [cost,grad] =  deepCost(weightsVector,net,trData,trPara);
%     %gradien checking 
% % %     numgrad = computeNumericalGradient(@(x)... 
% % %         deepCost(x,net,trData,trPara),weightsVector);
% % %     disp([numgrad grad]);
%     weightsVector = weightsVector - trPara.lr*grad;
%     net.cost = [net.cost;cost];
%     if(cost < trPara.performance)
%         [net] = vector2net(weightsVector,net);
%         net.para = trPara;
%         return
%     end
%     toc;
% end
[net] = vector2net(weightsVector,net);
net.para = trPara;
net.trainingTime = trainingTime;
end