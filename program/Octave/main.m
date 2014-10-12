%main script to call other ones
%
%Daniel Freitas & Alex Shkunov
%
%05.10.2014 thesis project 
%Reducing error rates in OCR
%Haaga-Helia University of Applied Sciences, Finland

clear ; close all; clc
disp('***Hello project-x***');

%% User defined parameters
trainSetFileName = '../train_1to3000.csv';
crossSetFileName = '../train_3001to4000.csv';
testSetFileName = '../train_4001to5000.csv';

%Data load
disp('Loading Training set...');
data = load(trainSetFileName);
disp('Loaded...');
[m, columnsAmount] = size(data);
n = columnsAmount-1;

fprintf('\nAmount of rows: %i \n', m);
y = data(:, 1) + 1; %In digits dataset 1st column are labels
% Label values are swaped for calculations, e.g. Zero = 1st class, One = 2nd class...
X = data (:, [2:1:n+1]); % pixels values are starts on second column

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Network parameters
input_layer_size = n; % NxN Input Images of Digits
hidden_layer_size = 50; % N hidden units
num_labels = 10; % 10 labels, from 1 to 10 

disp('Initializing Neural Network Parameters ...');
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters 
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Regularization parameter
lambda = 0;

%== Cost function with initial (random) weights
J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, 
	num_labels, X, y, lambda);

fprintf('\nCost at parameters initially (random): %f \n', J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%==Training Neural Network
disp('Training Neural Network...');

%options for optimisation
options = optimset('MaxIter', 50);

%Define that nnCostFunction should be optimised withn respect to first Parameter, e.g. Weights
costFunction = @(p) nnCostFunction(p, input_layer_size, 
			hidden_layer_size, num_labels, X, y, lambda);

%Run optimisation
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), 
	hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), 
	num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Basic evaluation: Trainig fit, test accuracy
disp('Basic evaluation....');

%Load test/cross-validation data
testData = load(testSetFileName);

test_y = testData(:, 1) + 1; % Swap values by one
test_X = testData(:, [2:1:n+1]); 

fprintf('\nTest-set Amount of rows: %i \n', rows(test_X));

predTrain = predict(Theta1, Theta2, X);
predTest = predict(Theta1, Theta2, test_X);

fprintf('\nTraining Fit: %2.2f %% ', mean(double(predTrain == y)) * 100);
fprintf('\nTest Accuracy: %2.2f %% \n', mean(double(predTest == test_y)) * 100);
