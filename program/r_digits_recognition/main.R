#main project script
#R Digits Recognition Neural Networks
#Daniel Freitas and Alex Shkunov
#
#Date: 23.10.2014
#Thesis: Haaga-Helia UAS

#Removes list from main memory
rm(list=ls())

#Import stuff
library(neuralnet)

#Global variables
digits_train <- "../train_full.csv"
hidden_layer_size <- 


#run actual stuff 
print("Starting Actual Stuff")

#loading into the main memory
digits_dataset <- read.csv(digits_train)
digits_train_set <- digits_dataset[1:25200, ]
digits_validation_cross_set <- digits_dataset[25201:33600, ]
digits_test_set <- digits_dataset[33601:42000, ]

