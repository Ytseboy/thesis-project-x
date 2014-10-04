#main script to call other ones
#
#Daniel Freitas & Alex Shkunov
#
#04.10.2014 thesis project 
#Reducing error rates in OCR
#Haaga-Helia University of Applied Sciences, Finland

#source needed scripts
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

d <- 0.09 # weights initialization helper
Theta1 <- 0 #should matrix with ranodm values
Theta2 <- 0 #should matrix with ranodm values

