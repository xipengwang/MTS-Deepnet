function numgrad = computeNumericalGradient(J, theta)
numgrad = zeros(size(theta));
% Hint: You will probably want to compute the elements of numgrad one at a time.
EPSILON = 1E-6;
numberOfElements = 20; %length(numgrad);
for i=[1:numberOfElements,length(numgrad)-10:length(numgrad)]
    temp = zeros(size(theta));
    temp(i) = 1;
    thetaPlus = theta + EPSILON*temp;
    thetaMinus = theta - EPSILON*temp;
    numgrad(i) = (J(thetaPlus) - J(thetaMinus))/(2*EPSILON);
end






%% ---------------------------------------------------------------
end
