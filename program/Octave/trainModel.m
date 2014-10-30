function [Theta1, Theta2, J_hist] = trainModel(X, y, initial_Theta1, initial_Theta2, hiddenUnits, lambda, iterations)

	[m, n] = size(X);

	%networkParameters
	input_layer_size = n; % NxN Input Images of Digits
	num_labels = length(unique(y)); % 10 labels, from 1 to 10 

	% Unroll parameters 
	initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

	%options for optimisation
	options = optimset('MaxIter', iterations);

	%Define that nnCostFunction should be optimised withn respect to first Parameter, e.g. Weights
	costFunction = @(p) nnCostFunction(p, input_layer_size, 
			hiddenUnits, num_labels, X, y, lambda);

	%Run optimisation
	[nn_params, J_hist] = fmincg(costFunction, initial_nn_params, options);

	% Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hiddenUnits * (input_layer_size + 1)), 
		hiddenUnits, (input_layer_size + 1));
	Theta2 = reshape(nn_params((1 + (hiddenUnits * (input_layer_size + 1))):end), 
		num_labels, (hiddenUnits + 1));

end