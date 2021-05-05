### Name: Xiaosong Fan
### Team B
### MSBX 5415 Final Project
### KC Housing Price

## Load dataset
rm(list = ls())
kc <- read.csv("kc_house_data.csv")

library(ipred)
library(ggplot2)
library(mlbench)
library(caret)
library(Metrics)


## Drop a few unnecessary columns
drop <- c("id", "date", "zipcode")
kc <- kc[,!names(kc) %in%drop]

## Create the formula used for model building
name <- names(kc)
formula <- as.formula(paste("price ~", paste(name[!name %in% "price"], collapse = " + ")))
formula


## Create 80/20 of train/test
set.seed(300)
index <- sample(1:nrow(kc),round(0.8*nrow(kc)))
train <- kc[index,]
test <- kc[-index,]


### Finding the optiaml number of trees =================================
# Assess 10-50 bagged trees
ntree <- 10:50

# Create empty vector to store OOB RMSE values
rmse <- vector(mode = "numeric", length = length(ntree))

for (i in seq_along(ntree)) {
  # reproducibility
  set.seed(123)
  
  # perform bagged model
  model <- bagging(formula = formula, data = train,
    coob = TRUE, nbagg = ntree[i]
  )
  # get OOB error
  rmse[i] <- model$err
}

plot(ntree, rmse, type = 'l', lwd = 2)
abline(v = 26, col = "red", lty = "dashed")



set.seed(123)

### Bagging =================================
## Building the bagging
kc.bagging <- bagging(formula, data = train, coob = TRUE)
kc.bagging

## Prediction results using bagging
bagging.pred <- predict(kc.bagging, newdata = test)

## create a data to combine the predicted and actual data
bagging.frame <- data.frame(cbind(actual_values = test$price, predicted_values = bagging.pred))


## Plot bagging
plot(x = bagging.frame$predicted_values, y = bagging.frame$actual_values,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs. Actual: Bagging Model w/ test data",
     col = "dodgerblue", pch = 20) + 
  grid() + 
  abline(0, 1, col = "darkorange", lwd = 2)

## Calculate accuracy for bagging
bagging.rss <- sum((bagging.frame$predicted_values - bagging.frame$actual_values)^2)
bagging.tss <- sum((bagging.frame$actual_values - mean(bagging.frame$actual_values))^2)
bagging.acc <- 1 - bagging.rss/bagging.tss
bagging.acc


## Calculate RMSE for the bagging method
bagging.rmse <- rmse(bagging.frame$actual_values, bagging.frame$predicted_values)
bagging.rmse


  
### Baggin with the optimal number of trees =================================
## Building the baggingv2
kc.baggingv2 <- bagging(formula = formula, data = train,coob = TRUE, nbagg = 26)
kc.baggingv2

## Prediction results using baggingv2
baggingv2.pred <- predict(kc.baggingv2, newdata = test)

## create a data to combine the predicted and actual data
baggingv2.frame <- data.frame(cbind(actual_values = test$price, predicted_values = baggingv2.pred))


## Plot baggingv2
plot(x = baggingv2.frame$predicted_values, y = baggingv2.frame$actual_values,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs. Actual: Optimal Bagging Model w/ test data",
     col = "dodgerblue", pch = 20) + 
  grid() + 
  abline(0, 1, col = "darkorange", lwd = 2)

## Calculate accuracy for baggingv2
baggingv2.rss <- sum((baggingv2.frame$predicted_values - baggingv2.frame$actual_values)^2)
baggingv2.tss <- sum((baggingv2.frame$actual_values - mean(baggingv2.frame$actual_values))^2)
baggingv2.acc <- 1 - baggingv2.rss/baggingv2.tss
baggingv2.acc


## Calculate RMSE for the baggingv2 
baggingv2.rmse <- rmse(baggingv2.frame$actual_values, baggingv2.frame$predicted_values)
baggingv2.rmse



### Results =================================
cat(" Accuracy for bagging is: ", bagging.acc, "\n",
    "Accuracy for bagging with optimal trees is: ", baggingv2.acc, "\n",
    "RMSE for bagging is: ", bagging.rmse, "\n",
    "RMSE for bagging with optimal trees is: ", baggingv2.rmse)



