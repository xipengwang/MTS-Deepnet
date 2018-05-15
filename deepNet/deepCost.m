%% Deep network training
function [cost,grad] = deepCost(vector,net,trData,trPara)
%% Parameters
%trData: data.positive(lane change events) and data.negative(lane keeping events)

%trPara.

%net

%% initial network
netStruct = trPara.netStruct;
%netStruct = [N  K2   K3    ... Kn   h  1...
%             M  L2   L3    ... Ln   0  0];
% even layer:   convolutional layer
% odd layer:    down-sample layer
active_func_option = trPara.active_func_option;
final_active_func_option = trPara.final_active_func_option;
layerNum = size(netStruct,2);
net = vector2net(vector,net);
N = netStruct(1,1);
M = netStruct(2,1);
%% random data
numOfPositive = size(trData.positive,2);
numOfNegative = size(trData.negative,2);
randIdx = randperm(numOfNegative + numOfPositive);
for i=1:(numOfNegative + numOfPositive)
    if(randIdx(i)<=numOfPositive)
        data{1,i} = trData.positive{1,randIdx(i)};
    else
        data{1,i} = trData.negative{1,randIdx(i)-numOfPositive};
    end
end

%% forward propagation
sampleN = size(data,2);
cost = 0;
for k = 1:sampleN
    tr_data = data{k}(:,2:M+1);
    tr_target = data{k}(1,1);
    net.layer{1}{1}.a = tr_data;
    net.layer{1}{1}.z = tr_data;
    for i = 2:layerNum-2
        if(mod(i,2)==0) %convolutional layer
            for j=1:length(net.layer{i})
                sectionNum = length(net.layer{i})/length(net.layer{i-1});
                s = net.layer{i}{j};
                s_pre = net.layer{i-1}{ceil(j/sectionNum)};
                s.z = valid_convolve_matrix(s.w,s_pre.a);
                s.z = s.z + ones(size(s.z,1),1)*s.b;
                s.a = fActive(s.z,active_func_option);
                net.layer{i}{j} = s;
            end
        else %down-sample layer
            for j=1:length(net.layer{i})
                sectionNum = length(net.layer{i})/length(net.layer{i-1});
                s = net.layer{i}{j};
                s_pre = net.layer{i-1}{ceil(j/sectionNum)};
                for t = 1:size(s.a,1)
                    s.z(t,:) = (s.w(:,:,t)*s_pre.a(t,:)' + s.b(:,:,t))';
                    s.a = fActive(s.z,active_func_option);
                end
                net.layer{i}{j} = s;
            end
        end
    end
    % final hidden layer
    i = i + 1;
    s = net.layer{i}{1};
    tmpInput = [];
    for j=1:length(net.layer{i-1})
        tmpInput = [tmpInput;net.layer{i-1}{j}.a];
    end
    s.z = s.w * tmpInput + s.b;
    clear tmpInput;
    s.a = fActive(s.z,active_func_option);
    net.layer{i}{1} = s;
    % final layer
    i = i + 1;
    s = net.layer{i}{1};
    s_pre = net.layer{i-1}{1};
    s.z = s.w * s_pre.a + s.b;
    s.a = fActive(s.z,final_active_func_option);
    net.layer{i}{1} = s;
    final_output = net.layer{layerNum}{1}.a;
    cost = cost + 0.5 *(final_output - tr_target)^2 ;
    %% backward propagation
    i = layerNum; %final layer
    s = net.layer{i}{1};
    s_pre = net.layer{i-1}{1};
    s.errorterm = -(tr_target - final_output).*fDeriv(s.z,final_active_func_option);
    s.w_delta = s.w_delta + s.errorterm * s_pre.a';
    s.b_delta = s.b_delta + s.errorterm;
    net.layer{i}{1} = s;
    s_pre.errorterm = s.w'*s.errorterm.*fDeriv(s_pre.z,active_func_option);
    net.layer{i-1}{1} = s_pre;
    i = layerNum-1; %final hidden layer
    s = net.layer{i}{1};
    tmpInput_a = [];
    tmpInput_z = [];
    for j=1:length(net.layer{i-1})
        tmpInput_a = [tmpInput_a;net.layer{i-1}{j}.a];
        tmpInput_z = [tmpInput_z;net.layer{i-1}{j}.z];
    end
    s.w_delta = s.w_delta + s.errorterm * tmpInput_a';
    s.b_delta = s.b_delta + s.errorterm;
    net.layer{i}{1} = s;
    clear tmpInput_a;
    tmpData = s.w'*s.errorterm.*fDeriv(tmpInput_z,active_func_option);
    clear tmpInput_z;
    for j=1:length(net.layer{i-1})
        s_pre = net.layer{i-1}{j};
        s_pre.errorterm = tmpData((j-1)*length(s_pre.errorterm)+1:j*length(s_pre.errorterm));
        net.layer{i-1}{j} = s_pre;
    end
    
    for i = layerNum-2:-1:2
        if(mod(i,2)==0) %convolutional layer
            for j=1:length(net.layer{i})
                sectionNum = length(net.layer{i})/length(net.layer{i-1});
                s = net.layer{i}{j};
                s_pre = net.layer{i-1}{ceil(j/sectionNum)};
                
                s.w_delta = s.w_delta + valid_convolve_matrix(s.errorterm,s_pre.a);
                s.b_delta = s.b_delta + sum(s.errorterm);
                
                net.layer{i}{j} = s;
                s_pre.errorterm = fully_convolve_matrix(s.errorterm,s.w).*fDeriv(s_pre.z,active_func_option);
                net.layer{i-1}{ceil(j/sectionNum)} = s_pre;
            end
        else %down-sample layer
            for j=1:length(net.layer{i})
                sectionNum = length(net.layer{i})/length(net.layer{i-1});
                s = net.layer{i}{j};
                s_pre = net.layer{i-1}{ceil(j/sectionNum)};
                for t=1:size(s.a,1)
                    s.w_delta(:,:,t) = s.w_delta(:,:,t) + s.errorterm(t,:)'*s_pre.a(t,:);
                    s.b_delta(:,:,t) = s.b_delta(:,:,t) + s.errorterm(t,:)';
                end
                net.layer{i}{j} = s;
                for t=1:size(s.a,1)
                    s_pre.errorterm(t,:) = [s.w(:,:,t)'*s.errorterm(t,:)'.*fDeriv(s_pre.z(t,:)',active_func_option)]';
                end
                net.layer{i-1}{ceil(j/sectionNum)} = s_pre;
            end
        end
    end
end
grad = [];
for i=2:size(net.layer,2)
    for j=1:size(net.layer{i})
        grad = [grad;net.layer{i}{j}.w_delta(:)];
        grad = [grad;net.layer{i}{j}.b_delta(:)];
    end
end
grad = grad/sampleN;
cost = cost/sampleN;
end