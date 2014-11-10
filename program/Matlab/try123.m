HU = (180:1:200);
resHU = zeros(length(HU), 3);

dataset_full = csvread('../reduced_14x14_fullNoHeaders.csv');
labels = dataset_full(:,1);

inputs = dataset_full(:,2:end);
inputs = inputs.';

targets = dummyvar(categorical(labels));
targets = targets.';

for i = 1:length(HU)

	fprintf('calibrationHU: %i \n', HU(i));

	%different amount of weights required for each (HU) model
    %Creating neural net
    net = patternnet(i);

    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.10;
    net.divideParam.testRatio = 0.10;

    net.trainParam.epochs = 450;
    net.trainParam.goal = 1e-5;
    net.performParam.regularization = 0.575;
    
    [net,tr] = train(net, inputs, targets);
    
    %Preparing for performance verification
    trainX = inputs(:,tr.trainInd);
    trainT = targets(:,tr.trainInd);
    testX = inputs(:,tr.testInd);
    testT = targets(:,tr.testInd);

	resHU(i, 1) = 1 - check(net(trainX),trainT);
	resHU(i, 2) = 1 - check(net(testX), testT);
	resHU(i, 3) = HU(i);

end

%% Plot
plot(resHU(:,3), resHU(:,1), resHU(:,3), resHU(:,2), 'r'); % Training & validation error over HU used for training
title('Hidden units optima');
xlabel('hidden units');
ylabel('Error %');
legend('Train err', 'Validation err');