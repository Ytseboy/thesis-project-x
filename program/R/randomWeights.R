randomWeights <- function(L_in, L_out){

	e <- 0.12 # weights initialization helper
	amount <- L_out * (L_in + 1)
	W <- matrix(runif(amount, 0.0, 1.0), ncol=L_in + 1) * (2*e) - e
}
