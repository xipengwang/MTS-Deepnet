function [out_net] = vector2net(vector,net)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
label = 1;
for i=2:size(net.layer,2)
    for j=1:size(net.layer{i})
        [w_row,w_col,w_3d]=size(net.layer{i}{j}.w);
        tmp = (net.layer{i}{j}.w - reshape(vector(label:label+w_row*w_col*w_3d-1),size(net.layer{i}{j}.w)));
        net.layer{i}{j}.w = reshape(vector(label:label+w_row*w_col*w_3d-1),size(net.layer{i}{j}.w));
        label = label + w_row*w_col*w_3d;
        [b_row,b_col,b_3d]=size(net.layer{i}{j}.b);
        net.layer{i}{j}.b = reshape(vector(label:label+b_row*b_col*b_3d-1),size(net.layer{i}{j}.b));
        label = label + b_row*b_col*b_3d;
    end
end
out_net = net;
end

