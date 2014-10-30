reduceStuff <- function(data, N, doWrite){
library(stats)
library(lattice)
summary(x <- princomp(data[,-1]))
loadings(x)

pca.plot <- xyplot(x$scores[,2] ~ x$scores[,1])
pca.plot$xlab <- "First Component"
pca.plot$ylab <- "Second Component"
pca.plot

p <- x$scores[,1:N]

  #Write an new N-d excel
  if(doWrite == 1){
    
    NewData <- cbind(data[,1], p)
    write.csv(NewData, paste("reduced_", N, "_rows_", nrow(p), ".csv"), row.names=FALSE)
  }
  else{
    return(p)
  }

}


