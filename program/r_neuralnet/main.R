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
source('classify.R')

#Global variables
digits_FileName <- "../train_headers_1to5000.csv"
hidden_layer_size <- 20
tr = 1 #Threshold for partial derevative for Performance control

#run actual stuff 
print("Starting Actual Stuff")

###Data preprocessing###
#loading into the main memory
digits_dataset <- read.csv(digits_FileName)
m <- nrow(digits_dataset)
print(paste("Data Loaded... ", m, " rows"))

#Label binary transformation
y <- digits_dataset[,1] + 1 # +1 for shift, so Zero is 1st class, One is 2nd class....
num_classes <- length(unique(y)) #amount of classes

I <- diag(num_classes) #Identity matrix
Y <- matrix(0, m, num_classes) #Binary matrix of the class

for(i in 1:m){
  Y[i,] <- I[y[i],]
}

#Column_Names and constructing the dataset with binary label
labelColumnNames <- paste("label", 0:9, sep = "")
featuresColumnNames <- colnames(digits_dataset)[-1]

#Calculation for 60/20/20
b1 <- m * 0.6
b2 <- b1 + m * 0.2

#Separate dataset into 60/20/20
digits_train_set <- digits_dataset[1:b1, ]  
digits_validation_set <- digits_dataset[(b1+1):b2, ]
digits_test_set <- digits_dataset[(b2+1):m, ]

print(paste("Train_set ", nrow(digits_train_set), " rows"))
print(paste("Test_set ", nrow(digits_test_set), " rows"))

#Binary dataset for Training the model
digits_train_binary <- cbind(Y[1:b1,], digits_train_set[,-1])
colnames(digits_train_binary) <- c(labelColumnNames, featuresColumnNames)

##Temporary, clean memory a bit
rm(digits_FileName, digits_dataset, m, y, num_classes, I, Y, i, b1, b2)

###Trainining model###
print(paste("Training Neural Network... ", hidden_layer_size, " Hidden units"))
f <- as.formula(paste(paste(labelColumnNames, collapse = "+"),
                      "~", 
                      paste(featuresColumnNames, collapse = "+")))
model <- neuralnet(f, data=digits_train_binary, hidden=hidden_layer_size, 
                   threshold = tr, lifesign="full",
                   lifesign.step = 10, linear.output = FALSE)

###Training fit and test accuracy
print("Model evaluation... ")

#Classify the data
classTrain <- classify(model, digits_train_set[,-1])
classTest <- classify(model, digits_test_set[,-1])
#Calculate error
trainFit <- round(mean(digits_train_set[,1] == classTrain), digits=2)
testAcc <- round(mean(digits_test_set[,1] == classTest), digits=2)

print(paste("Training fit:", trainFit*100, "%"))
print(paste("Test accuracy:", testAcc*100, "%"))

