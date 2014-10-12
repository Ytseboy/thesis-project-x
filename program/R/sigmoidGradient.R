sigmoidGradient <- function(z){
	g <- sigmoid(z)*(1 - sigmoid(z))
	return(g)
}