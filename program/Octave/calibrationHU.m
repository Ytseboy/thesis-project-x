%% Model Hidden units calibration
%% To be run after data loaded

% array of possible Hidden units amount
HU = [50:50:200];
%resultative matrix
resHU = zeros(length(HU), 3);

for i = 1:length(HU)

	fprintf('calibrationHU: %i', HU(i));

	%different amount of weights required for each (HU) model
	calInitial_Theta1 = randInitializeWeights(input_layer_size, HU(i));
	calInitial_Theta2 = randInitializeWeights(HU(i), num_labels);

	[calTheta1, calTheta2] = trainModel(X, y, calInitial_Theta1, calInitial_Theta2, HU(i), lambda, maxIter);

	resHU(i, 1) = 100 - assert(calTheta1, calTheta2, X, y);
	resHU(i, 2) = 100 - assert(calTheta1, calTheta2, val_X, val_y);
	resHU(i, 3) = HU(i);

endfor

%% Plot
plot(resHU(:,3), resHU(:,1), resHU(:,3), resHU(:,2), 'r'); % Training & validation error over HU used for training
title ("Hidden units optima");
xlabel ("hidden units");
ylabel ("Error %");
legend("Train err", "Validation err");
