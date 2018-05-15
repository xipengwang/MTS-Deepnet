function [ data ] = randomDownWeights(x,y,z)
%data three dimension matrix, it has z matrics with dimesion (x,y)

% http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
%var = 2/(n_in+n_out);

sigma = sqrt(2/(x+y));
mu = 0;
for i=1:z
    data(:,:,i)  = normrnd(mu,sigma,[x,y]);
end
end

