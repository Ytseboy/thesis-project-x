classify <- function(model, data){
  confidence <- predict(model, as.matrix(data))
  return(max.col(confidence)-1) #-1 for shifting back, 1st class -> Zero...
}