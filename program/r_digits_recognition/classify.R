classify <- function(model, data){
  confidence <- compute(model, data)$net.result
  return(max.col(confidence)-1) #-1 for shifting back, 1st class -> Zero...
}