#main script to call other ones
#
#Daniel Freitas & Alex Shkunov
#
#04.10.2014 thesis project 
#Reducing error rates in OCR
#Haaga-Helia University of Applied Sciences, Finland

#include
source('dataLoad.r')

#run actual stuff
print("***Hello project-x***")

#dataLoad
trainSet <- dataLoad("trainLoadTry.csv")
print("**Loaded**")

#X [features], y [label] in Digits dataset label is the first column
#n [amount of features], m [amount of tuples]
matrixSize <- dim(trainSet)
n <- matrixSize[2] - 1
m <- matrixSize[1]
print(paste("wow, ", m, " rows"))
y <- trainSet[,1]
X <- trainSet[,2:matrixSize[2]]

print("Press [enter] to continue")
readline()

#Network parameters, maybe to extract to own function/script?
input_layer_size <- n
hidden_layer_size <- 2
num_labels <- 10 # corresponding to amount of classes [0:9]

#random initialisation of network weights
d <- 0.09 # weights initialization helper
theta1amount <- hidden_layer_size * (input_layer_size + 1)
theta2amount <- num_labels * (hidden_layer_size + 1)

#runif(amount of numbers to generate, minimum, maximum)
Theta1 <- matrix(runif(theta1amount, 0.0, 1.0), ncol=input_layer_size + 1) * (2*d) - d 
Theta2 <- matrix(runif(theta2amount, 0.0, 1.0), ncol=hidden_layer_size + 1) * (2*d) - d

#regularisation parameter, will be needed later
lambda <- 0 

#call Cost function at initial stage
J <- 100 #TODO: separate script for cost function

print(paste("Cost with initial weghts = ", J))
print("Press [enter] to continue")
readline()

#Training ANN
print("Training NN...")

