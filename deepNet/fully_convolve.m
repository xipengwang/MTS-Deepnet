function [y] = fully_convolve(h,x)
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here
h=flipdim(h,1);
h_length = length(h);
x = [zeros(h_length-1,1);x;zeros(h_length-1,1)];
x_length = length(x);
y_length = x_length - h_length + 1;
y = zeros(y_length,1);
for i=1:y_length
   y(i) = h'*x(i:h_length+i-1); 
end


end

