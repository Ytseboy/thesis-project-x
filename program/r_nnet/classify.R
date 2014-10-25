classify <- function(model, data){
  
  r <- max.col(predict(model, as.matrix(data)))-1
  #-1 for shifting back, 1st class -> Zero...
  
  return(r) 
}