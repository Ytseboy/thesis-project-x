%% LearningCurve script to be run after learning
%%%
learningBoundary = 2000;
myDiv = 100;
%%%%
res = zeros((learningBoundary/myDiv), 3);

for i = 100:myDiv:learningBoundary

	fprintf('Learn: %i', i);

	[Theta1, Theta2, dummy] = trainModel(X(1:i,:), y(1:i), 
		initial_Theta1, initial_Theta2, hidden_layer_size, lambda, maxIter);

	res((i/myDiv), 1) = 100 - assert(Theta1, Theta2, X(1:i,:), y(1:i));
	res((i/myDiv), 2) = 100 - assert(Theta1, Theta2, val_X, val_y);
	res((i/myDiv), 3) = i;

endfor

%%Plot
plot(res(:,3), res(:,1), res(:,3), res(:,2), 'r'); % Training & validation error over data used for training
title ("Learning curve");
xlabel ("dataset size");
ylabel ("Error %");
legend("Train err", "Validation err");
