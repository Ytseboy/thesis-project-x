classify <- function(model, data){
  confidence <- compute(model, data)$net.result
  return(round(confidence))
}