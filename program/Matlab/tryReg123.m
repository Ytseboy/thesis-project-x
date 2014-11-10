Lam = (0.12:0.01:0.63);
resReg = zeros(length(Lam), 3);

dataset_full = csvread('../reduced_14x14_fullNoHeaders.csv');
labels = dataset_full(:,1);

inputs = dataset_full(:,2:end);
inputs = inputs.';

targets = dummyvar(categorical(labels));
targets = targets.';

for i = 1:length(Lam)

	fprintf('calibrationHU: %2.2f \n', Lam(i));

	%different amount of weights required for each (HU) model
    %Creating neural net
    net = patternnet(194);

    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.10;
    net.divideParam.testRatio = 0.20;

    net.trainParam.epochs = 450;
    net.trainParam.goal = 1e-5;
    net.performParam.regularization = Lam(i);
    
    [net,tr] = train(net, inputs, targets);
    
    %Preparing for performance verification
    trainX = inputs(:,tr.trainInd);
    trainT = targets(:,tr.trainInd);
    testX = inputs(:,tr.testInd);
    testT = targets(:,tr.testInd);

	resReg(i, 1) = 1 - check(net(trainX),trainT);
	resReg(i, 2) = 1 - check(net(testX), testT);
	resReg(i, 3) = Lam(i);

end

%% Plot
plot(resReg(:,3), resReg(:,1), resReg(:,3), resReg(:,2), 'r'); % Training & validation error over HU used for training
title('Lambda optima');
xlabel('L');
ylabel('Error %');
legend('Train err', 'Validation err');