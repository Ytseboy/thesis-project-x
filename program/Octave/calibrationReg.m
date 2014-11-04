%% Model Regularization calibration
%% To be run after data loaded

% array of possible Regularization values
calLambda = [0; 0.1; 0.3; 0.7]; %%; 1.9; 3.7; 5.0; 10];
%resultative matrix
resReg = zeros(length(calLambda), 3);

for i = 1:length(calLambda)

	fprintf('calibrationReg: %2.2f', calLambda(i));

	[calTheta1, calTheta2] = trainModel(X, y, initial_Theta1, initial_Theta2, hidden_layer_size, calLambda(i), maxIter);

	resReg(i, 1) = 100 - assert(calTheta1, calTheta2, X, y);
	resReg(i, 2) = 100 - assert(calTheta1, calTheta2, val_X, val_y);
	resReg(i, 3) = calLambda(i);

endfor

%% Plot
plot(resReg(:,3), resReg(:,1), resReg(:,3), resReg(:,2), 'r'); % Training & validation error over Lambda used for training
title ("Regularization optima");
xlabel ("lambda");
ylabel ("Error %");
legend("Train err", "Validation err");
