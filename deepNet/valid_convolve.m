function [y] = valid_convolve(h,x)
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here

h_length = length(h);
x_length = length(x);
assert(x_length>=h_length,'x is short than h');
y_length = x_length - h_length + 1;
y = zeros(y_length,1);
for i=1:y_length
   y(i) = h'*x(i:h_length+i-1); 
end

end

