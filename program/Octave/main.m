%main script to call other ones
%
%Daniel Freitas & Alex Shkunov
%
%30.10.2014 thesis project 
%Reducing error rates in OCR
%Haaga-Helia University of Applied Sciences, Finland

clear ; close all; clc
disp('***Hello project-x***');

%% Globals
datasetFileName = '../reduced_14x14_fullNoHeaders.csv';
hidden_layer_size = 850; % N hidden units
lambda = 0.3; % Regularization parameter
maxIter = 80; % Optimisation iterations

%%% Data processing%%%
%Data load
disp('Loading Training set...');
data = load(datasetFileName);
disp('Loaded... Processing...');
%Split in 60/20/20
[mFull, columnsAmountFull] = size(data);
c1 = mFull * 0.6; c2 = mFull * 0.2 + c1;
trainSet = data(1:c1,:);
valSet = data((c1+1):c2,:);
testSet = data((c2+1):end, :);

%Split into X and Y
[m, columnsAmount] = size(trainSet);
n = columnsAmount-1;

fprintf('\nAmount of rows: %i \n', m);
y = trainSet(:, 1) + 1; %In digits dataset 1st column are labels
% Label values are swaped for calculations, e.g. Zero = 1st class, One = 2nd class...
X = trainSet(:, [2:1:n+1]); % pixels values are starts on second column

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Network parameters
input_layer_size = n; % NxN Input Images of Digits
num_labels = length(unique(y)); % 10 labels, from 1 to 10 

disp('Initializing Neural Network Parameters ...');
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters 
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%== Cost function with initial (random) weights
J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, 
	num_labels, X, y, lambda);

fprintf('\nCost at parameters initially (random): %f \n', J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%==Training Neural Network
disp('Training Neural Network...');

[Theta1, Theta2, J_hist] = trainModel(X, y, initial_Theta1, initial_Theta2, hidden_layer_size, lambda, maxIter);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Basic evaluation: Trainig fit, test accuracy
disp('Basic evaluation....');

%Load test/cross-validation data
val_y = valSet(:, 1) + 1;
val_X = valSet(:, [2:1:n+1]);
test_y = testSet(:, 1) + 1; % Swap values by one
test_X = testSet(:, [2:1:n+1]); 

fprintf('\nTest-set Amount of rows: %i \n', rows(test_X));

fprintf('\nTraining Fit: %2.2f %% ', assert(Theta1, Theta2, X, y) );
fprintf('\nTest Accuracy: %2.2f %% \n', assert(Theta1, Theta2, test_X, test_y) );
