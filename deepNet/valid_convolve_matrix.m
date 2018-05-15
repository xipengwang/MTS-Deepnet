function [Y] = valid_convolve_matrix(H,X)
%UNTITLED18 Summary of this function goes here
%   Detailed explanation goes here
Y=[];
for i = 1:size(H,2)
   Y = [Y,valid_convolve(H(:,i),X(:,i))]; 
end


end

