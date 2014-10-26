#Try out the RSNNS package
library(RSNNS)

digits_FileName <- "../train_headers_1to5000.csv"
digits_dataset <- read.csv(digits_FileName)

X <- digits_dataset[,-1]
Y <- decodeClassLabels(digits_dataset[,1])

myData <- splitForTrainingAndTest(X, Y, ratio=0.50)
myData <- normTrainingAndTestSet(myData)

model <- mlp(myData$inputsTrain, myData$targetsTrain, size=2, learnFuncParams=c(0.1),
             maxit=50, inputsTest=myData$inputsTest, targetsTest=myData$targetsTest)


p_test <- max.col(predict(model, as.matrix(myData$inputsTest)))
round(mean(p_test == max.col(myData$targetsTest)))

p_fit <- max.col(predict(model, as.matrix(myData$inputsTrain)))
round(mean(p_test == max.col(myData$targetsTrain)))

#v0.2
plotIterativeError(model)