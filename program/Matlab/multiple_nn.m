%Initializing stuff
hidden_layer_unit = 194;
dataset_full = csvread('reduced_new.csv');
labels = dataset_full(:,1);

inputs = dataset_full(:,2:end);
inputs = inputs.';

targets = dummyvar(categorical(labels));
targets = targets.';

%Creating neural net
net = patternnet(hidden_layer_unit);

net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.1;

net.trainParam.epochs = 450;
net.trainParam.goal = 1e-5;
net.performParam.regularization = 0.575;

%Training net
numNN = 10;
nets = cell(1, numNN);
for i=1:numNN
    %[net, tr] = 
    [nets{i},tr] = train(net, inputs, targets);
end


%Preparing for performance verification
testX = inputs(:,tr.testInd);
testT = targets(:,tr.testInd);

perfs = zeros(1, numNN);
y2Total = 0;
for i=1:numNN
    neti = nets{i};
    y2 = neti(testX);
    perfs(i) = mse(neti, testT, y2);
    y2Total = y2Total + y2;
end
perfs;
y2AverageOutput = y2Total/numNN;
perfAverageOutput = mse(nets{1}, testT, y2AverageOutput); 

%Confusion Plot
%Uncomment the following lines to display plot
plotconfusion(testT, y2AverageOutput);
