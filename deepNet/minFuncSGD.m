function [opttheta,outCost] = minFuncSGD(funObj,theta,trData,options)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9


%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
        'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
m = length(trData.positive)+length(trData.negative); % training set size
% Setup for momentum
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));
%%======================================================================
%% SGD loop
it = 0;
costVector = [];
averageCost=[];
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = options.momentum;
        end;

        % get next randomly selected minibatch
        posK = 1;
        negK = 1;
        tr_data.positive = cell(1);
        tr_data.negative = cell(1);
        for i=s:s+minibatch-1
             if(rp(i)<=length(trData.positive))
                tr_data.positive{posK} = trData.positive{rp(i)};
                posK = posK + 1;
             else
                tr_data.negative{negK} = trData.negative{rp(i)-length(trData.positive)}; 
                negK = negK + 1;
             end
        end

		[cost,grad] = funObj(theta,tr_data);
		costVector = [costVector;cost];
        % Instructions: Add in the weighted velocity vector to the
        % gradient evaluated above scaled by the learning rate.
        % Then update the current weights theta according to the
        % sgd update rule
		
		%Vanilla update 
		%theta = theta- alpha*grad;
		
		%Momentum update
        velocity = mom*velocity - alpha*grad;
        theta = theta + velocity;
		
		%Nesterov Momentum
        %pre_velocity = velocity;
		%velocity = mom * velocity - alpha * grad;
		%x += -mom * pre_velocity + (1 + mom) * velocity;
		
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
    end;
	averageCost = [averageCost;mean(costVector)];
    % aneal learning rate by factor of two after each epoch
	if(mod(length(averageCost),10)=0)
		alpha = alpha/2.0;
	end
	if(length(averageCost)>10)
		if(std(averageCost(end-9:end))<1E-6)
			break;
		end
	end
end;

opttheta = theta;
outCost = cost;
end
