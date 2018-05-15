function [predictions,truth] = deepTest(teData,net)
netStruct = net.para.netStruct;
active_func_option = net.para.active_func_option;
final_active_func_option = net.para.final_active_func_option;
layerNum = size(netStruct,2);
N = netStruct(1,1);
M = netStruct(2,1);
numOfPositive = size(teData.positive,2);
numOfNegative = size(teData.negative,2);
randIdx = [1:numOfNegative + numOfPositive];
for i=1:(numOfNegative + numOfPositive)
    if(randIdx(i)<=numOfPositive)
        data{1,i} = teData.positive{1,randIdx(i)};
    else
        data{1,i} = teData.negative{1,randIdx(i)-numOfPositive};
    end
end
sampleN = size(data,2);
predictions = [];
truth = [];
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
                    s.z(t,:) = [s.w(:,:,t)*s_pre.a(t,:)' + s.b(:,:,t)]';
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
    final_output = double((final_output>=0.5));
    predictions = [predictions;final_output];
    truth = [truth;tr_target];
end


end