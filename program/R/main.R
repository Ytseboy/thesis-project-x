#main script to call other ones
#
#Daniel Freitas & Alex Shkunov
#
#05.10.2014 thesis project 
#Reducing error rates in OCR
#Haaga-Helia University of Applied Sciences, Finland

rm(list=ls()) #remove ALL objects 

#include
source('dataLoad.r')
source('randomWeights.r')
source('nnCostFunction.r')
source('sigmoid.r')
source('sigmoidGradient.r')

##Globals
trainSetFileName <- "../trainLoadTry.csv"
crossSetFileName <- "x"
testSetFileName <- "x"

#run actual stuff
print("***Hello project-x***")

#dataLoad
trainSet <- dataLoad(trainSetFileName)
print("**Loaded**")

#X [features], y [label] in Digits dataset label is the first column
#n [amount of features], m [amount of tuples]
matrixSize <- dim(trainSet)
n <- matrixSize[2] - 1
m <- matrixSize[1]
print(paste("wow, ", m, " rows"))
y <- trainSet[,1] + 1 # for further calculations [nnCostFunction], y values should be in range 1:10
X <- trainSet[,2:matrixSize[2]]

print("Press [enter] to continue")
readline()

#Network parameters, maybe to extract to own function/script?
input_layer_size <- n
hidden_layer_size <- 2
num_labels <- max(y) # corresponding to amount of classes [1:10]

#runif(amount of numbers to generate, minimum, maximum)
initialTheta1 <- randomWeights(input_layer_size, hidden_layer_size)
initialTheta2 <- randomWeights(hidden_layer_size, num_labels)

#regularisation parameter, will be needed later
lambda <- 0 

#call Cost function at initial stage
J <- nnCostFunction(initialTheta1, initialTheta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

print(paste("Cost with initial weghts = ", J))
print("Press [enter] to continue")
readline()

#Training ANN
print("Training NN...")
# TODO training ANN
