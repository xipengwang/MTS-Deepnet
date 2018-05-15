function [Y] = fully_convolve_matrix(H,X)
%UNTITLED17 Summary of this function goes here
%   Detailed explanation goes here
Y=[];
for i = 1:size(H,2)
   Y = [Y,fully_convolve(H(:,i),X(:,i))]; 
end
end

