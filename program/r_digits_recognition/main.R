#main project script
#R Digits Recognition Neural Networks
#Daniel Freitas and Alex Shkunov
#
#Date: 23.10.2014
#Thesis: Haaga-Helia UAS

#Removes objects from main memory
rm(list=ls())

#Import stuff
library(neuralnet)
source('classify.r')

#Global variables
digits_FileName <- "../train_headers_1to3000.csv"
hidden_layer_size <- 50
maxiter = 200 # Optimisation steps for Performance control

#run actual stuff 
print("Starting Actual Stuff")

#loading into the main memory
digits_dataset <- read.csv(digits_FileName)
print("Data Loaded...")

#Calculation for 60/20/20
m = nrow(digits_dataset)
b1 = m * 0.6
b2 = b1 + m * 0.2

#Separate dataset into 60/20/20
digits_train_set <- digits_dataset[1:b1, ]
digits_validation_set <- digits_dataset[(b1+1):b2, ]
digits_test_set <- digits_dataset[(b2+1):m, ]

#Trainining model
print("Training Neural Network...")
f <- as.formula(paste("label ~", paste(colnames(digits_train_set)[-1], collapse = " + ")))
model <- neuralnet(f, data=digits_train_set, hidden=hidden_layer_size, threshold = 10, stepmax = 1000)

##Training fit and test accuracy
print("Model evaluation...")

#Classify the data
classTrain <- classify(model, digits_train_set[,-1])
classTest <- classify(model, digits_test_set[,-1])
#Calculate error
trainFit <- round(mean(digits_train_set[1] == classTrain), digits=2)
testAcc <- round(mean(digits_test_set[1] == classTest), digits=2)

print(paste("Training fit:", trainFit*100, "%"))
print(paste("Test accuracy:", testAcc*100, "%"))

