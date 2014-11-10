%%% Run after training
kaggleTest = csvread('../reducedTestForKaggle(updated).csv');
kaggleTest = kaggleTest';

anSum = 0;

for i=1:numNN
    neti = nets{i};
    an = neti(kaggleTest);
    anSum = anSum + an;
end
anAvg = anSum/numNN;
A = resolveCIM(anAvg) -1;
A = [(1:28000)' A];

csvwrite('kaggleAnswer.csv',A)
disp('done Answer');