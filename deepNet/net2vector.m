function [vector] = net2vector(net)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%% update weights
vector = [];
for i=2:size(net.layer,2)
    for j=1:size(net.layer{i})
        vector = [vector;net.layer{i}{j}.w(:)];
        vector = [vector;net.layer{i}{j}.b(:)];
    end
end
end

