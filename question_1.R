#clear environment
rm(list=ls())

#read data and set up data sets
ridge_data <- read.csv("logit_ridge.csv", header = FALSE)
colnames(ridge_data) <- c("y","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","x18","x19","x20")
ridge_data$y <- as.numeric(ridge_data$y)
ridge_test <- ridge_data[1:10,]
ridge_train <- ridge_data[11:nrow(ridge_data),]

#initialize gradient ascent elements
alpha <- 10^-4
b_last <- matrix(ncol=21,nrow=1,data=1)
colnames(b_last) <- c("b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","b10","b11","b12","b13","b14","b15","b16","b17","b18","b19","b20")
b <- b_last
b_history <- b_last
err <- 100
eps <- 0.0001
lambda <- 1
grad <- matrix(data=0,ncol=1,nrow=21)
y_train <- as.matrix(as.numeric(ridge_train[,1]))
x_train <- as.matrix(ridge_train[,2:21])

#calculation of grad descent
while(err > eps) {
  e <- exp(b_last[,1]+b_last[,2]*x_train[,1]+b_last[,3]*x_train[,2]+b_last[,4]*x_train[,3]+b_last[,5]*x_train[,4]+b_last[,6]*x_train[,5]+b_last[,7]*x_train[,6]+b_last[,8]*x_train[,7]+b_last[,9]*x_train[,8]+b_last[,10]*x_train[,9]+b_last[,11]*x_train[,10]+b_last[,12]*x_train[,11]+b_last[,13]*x_train[,12]+b_last[,14]*x_train[,13]+b_last[,15]*x_train[,14]+b_last[,16]*x_train[,15]+b_last[,17]*x_train[,16]+b_last[,18]*x_train[,17]+b_last[,19]*x_train[,18]+b_last[,20]*x_train[,19]+b_last[,21]*x_train[,20])
  grad[1,] <- sum(y_train-e/(1+e))
  for(j in 2:21){
    grad[j,] <- sum(x_train[j-1]*(y_train))-sum(x_train[j-1]*e/(1+e))-2*lambda*b_last[,j]
 }
  b = b_last + alpha*t(grad)
  err = norm(b - b_last, type = "2")
  b_last <- b
  b_history <- rbind(b_history, b_last)
  print(err)
}

#while (err > eps) {
#  u = sum(x_train %*% b_last[2:21]) + b_last[1]
#  grad[1] = sum(y_train-exp(u)/(1+exp(u)))
#  for (j in 2:21){
#    grad[j] = sum(y_train-exp(u)/(1+exp(u))* x_train[,j-1]) -2*lambda*b_last[j]
#  }
#  b <- b_last + alpha*t(grad)
#  err = norm(b - b_last, type = "2")
#  b_last = b
#  b_history <- rbind(b_history, b_last)
#  print(err)
#}

#calculate prediction error
y_test <- as.matrix(as.numeric(ridge_test[,1]))
x_test <- as.matrix(ridge_test[,2:21])
y_prob <- matrix(ncol=1,nrow=10,data=0)
for(i in 1:10){
  y_prob[i] <- exp(b[,1] + x_test[i,]%*%b[,2:21])/(1+exp(b[,1] + x_test[i,]%*%b[,2:21]))
}
pred_err <- (y_test-y_prob)^2
mean(pred_err)