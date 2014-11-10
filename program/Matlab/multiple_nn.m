%Initializing stuff
hidden_layer_unit = 194;
dataset_full = csvread('../reduced_14x14_fullNoHeaders.csv');
labels = dataset_full(:,1);

inputs = dataset_full(:,2:end);
inputs = inputs.';

targets = dummyvar(categorical(labels));
targets = targets.';

%Creating neural net
net = patternnet(hidden_layer_unit);

net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.10;
net.divideParam.testRatio = 0.10;

net.trainParam.epochs = 450;
net.trainParam.goal = 1e-5;
net.performParam.regularization = 0.39;

%Training net
numNN = 50;
nets = cell(1, numNN);
for i=1:numNN
    fprintf('train model: %i \n', i);
    [nets{i},tr] = train(net, inputs, targets);
end

%Preparing for performance verification
trainX = inputs(:,tr.trainInd);
trainT = targets(:,tr.trainInd);
testX = inputs(:,tr.testInd);
testT = targets(:,tr.testInd);

perfs = zeros(1, numNN); fit = perfs;
y2Total = 0; f2Total = 0;
for i=1:numNN
    neti = nets{i};
    y2 = neti(testX);
    f2 = neti(trainX);
    
    fit(i) = check(f2,trainT);
    perfs(i) = check(y2, testT);
    
    f2Total = f2Total + f2;
    y2Total = y2Total + y2;
end
f2Avg = f2Total/numNN;
y2AverageOutput = y2Total/numNN;

fitAvg =  check(f2Avg, trainT);
perfAverageOutput = check(y2AverageOutput, testT); 

fprintf('\nfit %2.2f %%', fitAvg * 100);
fprintf('\ntest %2.2f %% \n', perfAverageOutput * 100);

%Confusion Plot
%Uncomment the following lines to display plot
%plotconfusion(testT, y2AverageOutput);
