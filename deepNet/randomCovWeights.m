function [data] = randomCovWeights(x,y)
% data is a matrix (x * y)

% http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
%var = 2/(n_in+n_out);

% sigma = sqrt(2/(x+y));
% mu = 0;
% data  = normrnd(mu,sigma,[x,y]);

sigma = sqrt(1/x);
mu = 0;
data = zeros(x,y);
for i=1:y
   data(:,i) =  normrnd(mu,sigma,[x,1]);
end
end

