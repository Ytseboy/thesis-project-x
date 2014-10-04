#data load here for loading training data

dataLoad <- function(fileName){
	print("**Loading Data**")

	mydata = as.matrix(read.csv(fileName, header = FALSE))
	return(mydata)
}