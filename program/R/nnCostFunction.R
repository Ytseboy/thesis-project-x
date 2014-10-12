##neural netwrok Cost function [J]
nnCostFunction <- function(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda){
	
	m <- dim(X)[1]

	J <- 0
	Theta1_grad <- matrix(0, dim(Theta1)[1], dim(Theta1)[2])
	Theta2_grad <- matrix(0, dim(Theta2)[1], dim(Theta2)[2])

	#Feedforward propagation
	#Binary-Matrix for labels e.g. if y = 2 --> second column of Y = 1, other nine are false
	I <- diag(num_labels) #Identity matrix
	Y <- matrix(0, m, num_labels)

	for(i in 1:m){
		Y[i,] <- I[y[i],]
	}

	#cbind(firstMatrix, secondMatrix) ==> combined
	#matirx(1,m,1) is used for adding bias
	A1 <- cbind(matrix(1, m, 1), X) # first layer result
	Z2 <- A1 %*% t(Theta1)
	A2 <- cbind(matrix(1, dim(Z2)[1], 1), sigmoid(Z2)) #second layer result
	Z3 <- A2 %*% t(Theta2)
	H <- A3 <- sigmoid(Z3) # hypothesis

	reg <- (lambda/(2*m))*(sum(sum(Theta1(, -1)^2, 2)) + sum(sum(Theta2(,-1)^2, 2)));

	J <- (1/m)*sum(sum((-Y)*log(H) - (1 - Y) * log(1-H), 2))
	J <- J + reg

	#backpropagation
	Sigma3 <- A3 - Y
	Sigma2 <- (Sigma3 %*% Theta2 * sigmoidGradient(cbind(matrix(1, dim(Z2)[1], 1), Z2)))[,-1]

	Delta_1 <- t(Sigma2) %*% A1
	Delta_2 <- t(Sigma3) %*% A2

	Theta1_grad = Delta_1/m + (lambda/m)* cbind(matrix(0, dim(Theta1)[1], 1), Theta1(, -1))
	Theta2_grad = Delta_2/m + (lambda/m)* cbind(matrix(0, dim(Theta2)[1], 1), Theta2(, -1))

	return(J)

}