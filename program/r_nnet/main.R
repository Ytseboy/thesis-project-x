#main project script
#R Digits Recognition Neural Networks
#Daniel Freitas and Alex Shkunov
#
#Date: 23.10.2014
#Thesis: Haaga-Helia UAS

#Removes objects from main memory
rm(list=ls())

#Import stuff
library(nnet)
source('classify.R')

#Global variables
digits_FileName <- "../train_headers_1to5000.csv"
hidden_layer_size <- 50
maxiter <- 2000

#run actual stuff 
print("Starting Actual Stuff")

###Data preprocessing###
#loading into the main memory
digits_dataset <- as.matrix(read.csv(digits_FileName))
m <- nrow(digits_dataset)
print(paste("Data Loaded... ", m, " rows"))

#Calculation for 60/20/20
b1 <- m * 0.6
b2 <- b1 + m * 0.2

#Separate dataset into 60/20/20
digits_train_set <- digits_dataset[1:b1, ]  
digits_validation_set <- digits_dataset[(b1+1):b2, ]
digits_test_set <- digits_dataset[(b2+1):m, ]

print(paste("Train_set ", nrow(digits_train_set), " rows"))
print(paste("Test_set ", nrow(digits_test_set), " rows"))

#separate label and features for Train set
y <- digits_train_set[,1]
Y <- class.ind(y)
X <- digits_train_set[,-1]

###Trainining model###
print(paste("Training Neural Network... ", hidden_layer_size, " Hidden units"))
model <- nnet(x = X, y = Y, size = hidden_layer_size, maxit = maxiter, MaxNWts= 1000000)

###Training fit and test accuracy
print("Model basic performance evaluation... ")

#Classify the data
classTrain <- classify(model, digits_train_set[,-1])
classTest <- classify(model, digits_test_set[,-1])
#Calculate error
trainFit <- round(mean(digits_train_set[,1] == classTrain), digits=2)
testAcc <- round(mean(digits_test_set[,1] == classTest), digits=2)

print(paste("Training fit:", trainFit*100, "%"))
print(paste("Test accuracy:", testAcc*100, "%"))
